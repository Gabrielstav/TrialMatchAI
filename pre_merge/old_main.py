from __future__ import annotations

# Set multiprocessing start method to 'spawn' before any CUDA initialization,
# this is required for vLLM when CUDA may already be initialized.
# See: https://docs.vllm.ai/en/latest/design/multiprocessing.html
import multiprocessing
import os

if __name__ == "__main__":
    # only set in main process to avoid issues with spawn workers
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

# also set via environment variable for vLLM's internal multiprocessing
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from typing import Dict, List, Optional

import torch

from elasticsearch import Elasticsearch

from .config.config_loader import load_config
from .models.embedding.query_embedder import QueryEmbedder
from .models.embedding.sentence_embedder import SecondLevelSentenceEmbedder
from .models.llm.llm_loader import load_model_and_tokenizer
from .models.llm.llm_reranker import LLMReranker
from .pipeline.cot_reasoning import BatchTrialProcessor
from .pipeline.phenopacket_processor import process_phenopacket
from .pipeline.trial_ranker import (
    load_trial_data,
    rank_trials,
    save_ranked_trials,
)
from .pipeline.trial_search.first_level_search import ClinicalTrialSearch
from .pipeline.trial_search.second_level_search import SecondStageRetriever
from .services.biomedner_service import initialize_biomedner_services
from .utils.file_utils import (
    create_directory,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from .utils.logging_config import setup_logging

logger = setup_logging()


def log_gpu_memory_state(label: str = "") -> None:
    """Log GPU memory state and nvidia-smi output for debugging."""
    if not torch.cuda.is_available():
        return

    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    logger.info(
        f"[GPU {label}] free={free / 2**30:.2f}GB, total={total / 2**30:.2f}GB, "
        f"allocated={allocated / 2**30:.2f}GB, reserved={reserved / 2**30:.2f}GB"
    )

    # Also run nvidia-smi for process-level visibility
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            logger.info(f"[GPU processes]\n{result.stdout.strip()}")
    except Exception as e:
        logger.debug(f"nvidia-smi query failed: {e}")


def log_live_cuda_tensors(limit: int = 25) -> None:
    """Log live CUDA tensors to help diagnose memory not being freed.

    Helps distinguish between 'live refs' vs 'allocator pinned' scenarios.
    If this prints non-trivial tensors after cleanup, references still exist.
    """
    if not torch.cuda.is_available():
        return

    import gc

    items = []
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                n = o.numel() * o.element_size()
                items.append((n, tuple(o.shape), str(o.dtype), type(o).__name__))
        except Exception:
            pass

    items.sort(reverse=True, key=lambda x: x[0])
    logger.info(f"[CUDA] live cuda tensors={len(items)}")
    for n, shape, dtype, typ in items[:limit]:
        logger.info(
            f"[CUDA] {n / 2**20:7.2f} MiB  {typ:<18s} shape={shape} dtype={dtype}"
        )


def cleanup_gpu_memory(*objs) -> None:
    """Break CUDA ties by moving models to CPU.

    NOTE: This function only breaks CUDA ties. The caller MUST set their
    references to None after calling this, then run gc.collect() etc.
    """
    import gc

    log_gpu_memory_state("before cleanup")

    for obj in objs:
        if obj is None:
            continue
        # Handle wrapped models (like LLMReranker that has .model attribute)
        inner = getattr(obj, "model", None)
        for m in (inner, obj):
            if m is None:
                continue
            try:
                # Prefer to_empty (avoids CPU memory spike) if available
                if hasattr(m, "to_empty"):
                    m.to_empty(device="cpu")
                elif hasattr(m, "to"):
                    m.to("cpu")
            except Exception:
                pass

    gc.collect()
    gc.collect()  # Run twice to handle cyclic references

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    log_gpu_memory_state("after cleanup")
    log_live_cuda_tensors()


def run_first_level_search(
    keywords: Dict,
    output_folder: str,
    patient_info: Dict,
    bio_med_ner,
    embedder: QueryEmbedder,
    config: Dict,
    es_client: Elasticsearch,
) -> Optional[tuple]:
    main_conditions = keywords.get("main_conditions", [])
    other_conditions = keywords.get("other_conditions", [])
    expanded_sentences = keywords.get("expanded_sentences", [])

    if not main_conditions:
        logger.error("No main_conditions found in keywords.")
        return None

    condition = main_conditions[0]
    age = patient_info.get("age", "all")
    sex = patient_info.get("gender", "all")
    overall_status = "All"

    index_name = config["elasticsearch"]["index_trials"]
    cts = ClinicalTrialSearch(es_client, embedder, index_name, bio_med_ner)

    # Get synonyms and expand main conditions
    synonyms = cts.get_synonyms(condition.lower().strip())
    main_conditions.extend(synonyms[:5])

    search_size = config["search"].get("max_trials_first_level", 300)
    search_mode = config["search"].get("search_mode", "hybrid")
    trials, scores = cts.search_trials(
        condition=condition,
        age_input=age,
        sex=sex,
        overall_status=overall_status,
        size=search_size,
        pre_selected_nct_ids=None,
        synonyms=main_conditions,
        other_conditions=other_conditions,
        vector_score_threshold=config["search"]["vector_score_threshold"],
        search_mode=search_mode,
    )

    nct_ids = [trial.get("nct_id") for trial in trials if trial.get("nct_id")]
    first_level_scores = {
        trial.get("nct_id"): score
        for trial, score in zip(trials, scores)
        if trial.get("nct_id")
    }

    write_text_file([str(nid) for nid in nct_ids], f"{output_folder}/nct_ids.txt")
    write_json_file(first_level_scores, f"{output_folder}/first_level_scores.json")

    logger.info(f"First-level search complete: {len(nct_ids)} trial IDs saved.")
    return (
        nct_ids,
        main_conditions,
        other_conditions,
        expanded_sentences,
        first_level_scores,
    )


def run_second_level_search(
    output_folder: str,
    nct_ids: List[str],
    main_conditions: List[str],
    other_conditions: List[str],
    expanded_sentences: List[str],
    gemma_retriever: SecondStageRetriever,
    first_level_scores: Dict,
    config: Dict,
) -> tuple:
    queries = list(set(main_conditions + other_conditions + expanded_sentences))[:10]
    logger.info(f"Running second-level retrieval with {len(queries)} queries ...")

    # Add synonyms for second level
    if queries:
        synonyms = gemma_retriever.get_synonyms(queries[0])
        queries.extend(synonyms[:3])

    top_n = min(len(nct_ids), config["search"].get("max_trials_second_level", 100))
    second_level_results = gemma_retriever.retrieve_and_rank(
        queries, nct_ids, top_n=top_n
    )

    combined_scores = {}
    for trial in second_level_results:
        trial_id = trial["nct_id"]
        second_score = trial["score"]
        first_score = first_level_scores.get(trial_id, 0)
        combined_scores[trial_id] = first_score + second_score

    sorted_trials = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    num_top = max(1, min(len(sorted_trials) // 3, top_n))
    semi_final_trials = sorted_trials[:num_top]

    top_trials_path = f"{output_folder}/top_trials.txt"
    write_text_file([trial_id for trial_id, _ in semi_final_trials], top_trials_path)

    logger.info("Second-level retrieval and ranking complete. Top trials saved.")
    return semi_final_trials, top_trials_path


def run_rag_processing(
    output_folder: str,
    top_trials_file: str,
    patient_info: Dict,
    model,
    tokenizer,
    config: Dict,
):
    top_trials = read_text_file(top_trials_file)
    if not top_trials:
        logger.error("No top trials available for RAG processing.")
        return

    top_trials = top_trials[: config["rag"].get("max_trials_rag", 20)]
    patient_profile = patient_info.get("split_raw_description", [])
    if not patient_profile:
        logger.error("No patient profile available for RAG processing.")
        return

    # Check if vLLM backend is configured
    cot_backend = config.get("cot_backend", "default")
    use_vllm = cot_backend == "vllm"

    if use_vllm:
        logger.info("Using vLLM backend for CoT reasoning")

        # Lazy import vLLM modules (only available on Linux with CUDA)
        from .models.llm.vllm_loader import load_vllm_engine
        from .pipeline.cot_reasoning_vllm import BatchTrialProcessorVLLM

        # Load vLLM configuration
        vllm_cfg = config.get("vllm", {})

        # Load vLLM engine
        vllm_engine, vllm_tokenizer, lora_request = load_vllm_engine(
            model_config=config.get("model", {}),
            vllm_cfg=vllm_cfg,
        )

        # Create vLLM processor
        rag_processor = BatchTrialProcessorVLLM(
            llm=vllm_engine,  # type: ignore
            tokenizer=vllm_tokenizer,
            batch_size=vllm_cfg.get("batch_size", 16),
            use_cot=config.get("use_cot_reasoning", True),
            max_new_tokens=vllm_cfg.get("max_new_tokens", 5000),
            temperature=vllm_cfg.get("temperature", 0.0),
            top_p=vllm_cfg.get("top_p", 1.0),
            seed=vllm_cfg.get("seed", 1234),
            length_bucket=vllm_cfg.get("length_bucket", True),
            lora_request=lora_request,
        )
    else:
        logger.info("Using default (HuggingFace) backend for CoT reasoning")

        batch_size = min(config["rag"]["batch_size"] * 2, 8)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Get system role support flag from config (defaults to True for HPC with Phi-4)
        supports_system_role = config.get("model", {}).get("supports_system_role", True)

        rag_processor = BatchTrialProcessor(
            model,
            tokenizer,
            device=config["global"]["device"],
            batch_size=batch_size,
            supports_system_role=supports_system_role,
        )

    rag_processor.process_trials(
        nct_ids=top_trials,
        json_folder=config["paths"]["trials_json_folder"],
        output_folder=output_folder,
        patient_profile=patient_profile,
    )
    write_json_file({"status": "done"}, f"{output_folder}/rag_output.json")
    logger.info("RAG-based trial matching complete.")


def main_pipeline():
    logger.info("Starting TrialMatchAI pipeline...")
    config = load_config()
    create_directory(config["paths"]["output_dir"])

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

    initialize_biomedner_services(config)

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantization_config.*", category=UserWarning
        )
        model, tokenizer = load_model_and_tokenizer(
            config["model"], config["global"]["device"]
        )

    if tokenizer.pad_token is None:  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore
        if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:  # type: ignore
            model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore

    if config["global"]["device"] != "cpu" and torch.cuda.is_available():
        model = model.half()  # type: ignore

    # Initialize components
    first_level_embedder = QueryEmbedder(model_name=config["embedder"]["model_name"])
    second_level_embedder = SecondLevelSentenceEmbedder(
        model_name=config["embedder"]["model_name"]
    )

    # Initialize BioMedNER only if enabled
    bio_med_ner = None
    if config.get("bio_med_ner", {}).get("enabled", True):
        from ..Parser.biomedner_engine import BioMedNER

        # Filter out 'enabled' key before passing to BioMedNER constructor
        ner_config = {k: v for k, v in config["bio_med_ner"].items() if k != "enabled"}
        bio_med_ner = BioMedNER(**ner_config)
    else:
        logger.info("BioMedNER disabled - synonym expansion will be skipped")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantization_config.*", category=UserWarning
        )
        llm_reranker = LLMReranker(
            model_path=config["model"]["reranker_model_path"],
            adapter_path=config["model"]["reranker_adapter_path"],
            device=config["global"]["device"],
            batch_size=config["rag"]["batch_size"] * 2,
        )

    # Build ES client - auth/certs only when configured (disabled on HPC)
    es_user = config["elasticsearch"].get("username")
    es_pass = config["elasticsearch"].get("password")
    ca_certs = config["paths"].get("docker_certs")

    es_client = Elasticsearch(
        hosts=[config["elasticsearch"]["host"]],
        basic_auth=(es_user, es_pass) if es_user and es_pass else None,
        ca_certs=ca_certs if ca_certs else None,
        verify_certs=config["elasticsearch"].get("verify_certs", True),
        request_timeout=config["elasticsearch"]["request_timeout"],
        retry_on_timeout=config["elasticsearch"]["retry_on_timeout"],
    )

    gemma_retriever = SecondStageRetriever(
        es_client=es_client,
        llm_reranker=llm_reranker,
        embedder=second_level_embedder,
        index_name=config["elasticsearch"]["index_trials_eligibility"],
        bio_med_ner=bio_med_ner,
        search_mode=config["search"].get("search_mode", "hybrid"),
    )

    # Process phenopackets in two phases for GPU memory efficiency
    patient_folder = config["paths"]["patients_dir"]
    phenopacket_files = [f for f in os.listdir(patient_folder) if f.endswith(".json")]

    # Check if using vLLM backend (requires two-phase processing)
    use_vllm = config.get("cot_backend") == "vllm"

    # Store intermediate results for Phase 2
    patient_results: Dict[str, Dict] = {}

    # ==========================================================================
    # PHASE 1: Process ALL patients through search pipeline (HuggingFace models)
    # ==========================================================================
    logger.info(
        f"Phase 1: Processing {len(phenopacket_files)} patient(s) through search pipeline..."
    )

    for phenopacket_file in phenopacket_files:
        patient_id = phenopacket_file.split(".")[0]
        output_folder = f"{config['paths']['output_dir']}/{patient_id}"
        create_directory(output_folder)

        input_file = f"{patient_folder}/{phenopacket_file}"
        output_file = f"{output_folder}/keywords.json"

        # Get system role support flag from config (defaults to True for HPC with Phi-4)
        supports_system_role = config.get("model", {}).get("supports_system_role", True)

        with torch.no_grad():
            process_phenopacket(
                input_file,
                output_file,
                model=model,
                tokenizer=tokenizer,
                supports_system_role=supports_system_role,
            )

        keywords = read_json_file(output_file)
        patient_info = read_json_file(input_file)
        patient_info["split_raw_description"] = keywords.get("expanded_sentences", [])

        # Run search pipeline
        with torch.no_grad():
            result = run_first_level_search(
                keywords,
                output_folder,
                patient_info,
                bio_med_ner,
                first_level_embedder,
                config,
                es_client,
            )
        if not result:
            logger.error(f"First-level search failed for {patient_id}")
            continue

        (
            nct_ids,
            main_conditions,
            other_conditions,
            expanded_sentences,
            first_level_scores,
        ) = result

        with torch.no_grad():
            _, top_trials_path = run_second_level_search(
                output_folder,
                nct_ids,
                main_conditions,
                other_conditions,
                expanded_sentences,
                gemma_retriever,
                first_level_scores,
                config,
            )

        # Store results for Phase 2
        patient_results[patient_id] = {
            "output_folder": output_folder,
            "top_trials_path": top_trials_path,
            "patient_info": patient_info,
        }
        logger.info(f"Phase 1 complete for patient {patient_id}")

    # ==========================================================================
    # GPU MEMORY CLEANUP (between phases, only if using vLLM)
    # ==========================================================================
    if use_vllm and patient_results:
        import gc

        logger.info("Phase 1 complete. Cleaning up GPU memory for vLLM...")
        cleanup_gpu_memory(
            model,
            llm_reranker,
            gemma_retriever,
            first_level_embedder,
            second_level_embedder,
        )

        # Caller-side: explicitly drop all references
        model = None  # type: ignore
        tokenizer = None  # type: ignore
        llm_reranker = None  # type: ignore
        gemma_retriever = None  # type: ignore
        first_level_embedder = None  # type: ignore
        second_level_embedder = None  # type: ignore

        # Final cleanup after dropping references
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

        # Log final state before vLLM init
        log_gpu_memory_state("after reference drop")
        log_live_cuda_tensors()

    # ==========================================================================
    # PHASE 2: RAG/CoT processing for ALL patients
    # ==========================================================================
    logger.info(f"Phase 2: Running RAG processing for {len(patient_results)} patient(s)...")

    for patient_id, data in patient_results.items():
        with torch.no_grad():
            run_rag_processing(
                data["output_folder"],
                data["top_trials_path"],
                data["patient_info"],
                model,
                tokenizer,
                config,
            )

        # Final ranking
        trial_data = load_trial_data(data["output_folder"])
        ranked_trials = rank_trials(trial_data)
        save_ranked_trials(ranked_trials, f"{data['output_folder']}/ranked_trials.json")

        logger.info(f"Pipeline completed for patient {patient_id}")


if __name__ == "__main__":
    main_pipeline()
