from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from Parser.biomedner_engine import BioMedNER

from elasticsearch import Elasticsearch

from Matcher.config.config_loader import load_config
from Matcher.models.embedding.text_embedder import TextEmbedder, TextEmbedderConfig
from Matcher.models.llm.llm_loader import load_model_and_tokenizer
from Matcher.models.llm.llm_reranker import LLMReranker
from Matcher.models.llm.vllm_loader import load_vllm_engine
from Matcher.pipeline.cot_reasoning import BatchTrialProcessor
from Matcher.pipeline.cot_reasoning_vllm import BatchTrialProcessorVLLM
from Matcher.pipeline.phenopacket_processor import process_phenopacket
from Matcher.pipeline.trial_ranker import (
    load_trial_data,
    rank_trials,
    save_ranked_trials,
)
from Matcher.pipeline.trial_search.first_level_search import ClinicalTrialSearch
from Matcher.pipeline.trial_search.second_level_search import SecondStageRetriever
from Matcher.services.biomedner_service import initialize_biomedner_services
from Matcher.services.elasticsearch_service import ensure_elasticsearch
from Matcher.utils.file_utils import (
    create_directory,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from Matcher.schemas.phenopacket import Keywords, Phenopacket
from Matcher.utils.logging_config import reset_request_id, set_request_id, setup_logging
from Matcher.utils.timing import log_timing
from Matcher.utils.gpu_log import (
    log_gpu_memory_state,
    log_live_cuda_tensors,
    cleanup_gpu_memory,
)

# set multiprocessing start method to "spawn" before CUDA initialization,
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

logger = setup_logging(__name__)


def run_first_level_search(
    keywords: Dict,
    output_folder: str,
    patient_info: Dict,
    bio_med_ner,
    embedder: TextEmbedder,
    config: Dict,
    es_client: Elasticsearch,
) -> Optional[Tuple]:
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
) -> Tuple:
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
        supports_system_role = config.get("model", {}).get("base_model_supports_system_role", True)

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
    paths = config["paths"]
    create_directory(paths["output_dir"])

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
    embedder_cfg = config.get("embedder", {})
    embedder = TextEmbedder(
        TextEmbedderConfig(
            model_name=embedder_cfg.get("model_name", "BAAI/bge-m3"),
            pooling=embedder_cfg.get("pooling", "mean"),
            max_length=embedder_cfg.get("max_length", 512),
            batch_size=embedder_cfg.get("batch_size", 32),
            use_gpu=embedder_cfg.get("use_gpu", True),
            use_fp16=embedder_cfg.get("use_fp16", False),
            normalize=embedder_cfg.get("normalize", True),
        )
    )
    # Initialize BioMedNER only if enabled
    bio_med_ner = None
    if config.get("bio_med_ner", {}).get("enabled", True):
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
    ca_certs = paths.get("docker_certs")

    es_client = Elasticsearch(
        hosts=[config["elasticsearch"]["host"]],
        basic_auth=(es_user, es_pass) if es_user and es_pass else None,
        ca_certs=ca_certs if ca_certs else None,
        verify_certs=config["elasticsearch"].get("verify_certs", True),
        request_timeout=config["elasticsearch"]["request_timeout"],
        retry_on_timeout=config["elasticsearch"]["retry_on_timeout"],
    )
    if not ensure_elasticsearch(es_client, config):
        return

    gemma_retriever = SecondStageRetriever(
        es_client=es_client,
        llm_reranker=llm_reranker,
        embedder=embedder,
        index_name=config["elasticsearch"]["index_trials_eligibility"],
        bio_med_ner=bio_med_ner,
    )

    # Process phenopackets in two phases for GPU memory efficiency
    patient_folder = Path(paths["patients_dir"])
    if not patient_folder.exists():
        logger.error("Patients folder not found: %s", patient_folder)
        return
    phenopacket_files = sorted(
        [p for p in patient_folder.iterdir() if p.suffix == ".json"]
    )
    if not phenopacket_files:
        logger.warning("No patient files found in %s", patient_folder)
        return

    # GPU config - user sets based on their hardware (V100 needs cleanup, A100 doesn't)
    gpu_cfg = config.get("gpu", {})
    cleanup_between_phases = gpu_cfg.get("cleanup_between_phases", False)
    gpu_logging = gpu_cfg.get("logging", False)

    # Store intermediate results for Phase 2
    patient_results: Dict[str, Dict] = {}

    # ==========================================================================
    # PHASE 1: Process ALL patients through search pipeline
    # ==========================================================================
    logger.info(
        f"Phase 1: Processing {len(phenopacket_files)} patient(s) through search pipeline..."
    )

    for phenopacket_path in phenopacket_files:
        patient_id = phenopacket_path.stem
        token = set_request_id(patient_id)
        output_folder = Path(paths["output_dir"]) / patient_id
        create_directory(str(output_folder))

        input_file = str(phenopacket_path)
        output_file = str(output_folder / "keywords.json")

        try:
            with log_timing(logger, "Phenopacket processing"):
                with torch.no_grad():
                    process_phenopacket(
                        input_file, output_file, model=model, tokenizer=tokenizer
                    )

            keywords = Keywords.model_validate(read_json_file(output_file)).model_dump()
            patient_info = Phenopacket.model_validate(
                read_json_file(input_file)
            ).model_dump()
            patient_info["split_raw_description"] = keywords.get(
                "expanded_sentences", []
            )

            # Run search pipeline
            with log_timing(logger, "First-level search"):
                with torch.no_grad():
                    result = run_first_level_search(
                        keywords,
                        str(output_folder),
                        patient_info,
                        bio_med_ner,
                        embedder,
                        config,
                        es_client,
                    )
            if not result:
                logger.error("First-level search failed for %s", patient_id)
                continue

            (
                nct_ids,
                main_conditions,
                other_conditions,
                expanded_sentences,
                first_level_scores,
            ) = result

            with log_timing(logger, "Second-level search"):
                with torch.no_grad():
                    _, top_trials_path = run_second_level_search(
                        str(output_folder),
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
                "output_folder": str(output_folder),
                "top_trials_path": top_trials_path,
                "patient_info": patient_info,
            }
            logger.info(f"Phase 1 complete for patient {patient_id}")

        except Exception:  # noqa
            logger.exception("Phase 1 failed for patient %s", patient_id)
            continue
        finally:
            reset_request_id(token)

    # ==========================================================================
    # GPU MEMORY CLEANUP (between phases, if configured)
    # ==========================================================================
    if cleanup_between_phases and patient_results:
        import gc

        logger.info("Phase 1 complete. Cleaning up GPU memory...")

        if gpu_logging:
            log_gpu_memory_state("before cleanup")

        cleanup_gpu_memory(model, llm_reranker, gemma_retriever, embedder)

        # Explicitly drop all references
        model = None  # type: ignore
        tokenizer = None  # type: ignore
        llm_reranker = None  # type: ignore
        gemma_retriever = None  # type: ignore
        embedder = None  # type: ignore

        # Final cleanup after dropping references
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        if gpu_logging:
            log_gpu_memory_state("after cleanup")
            log_live_cuda_tensors()

    # ==========================================================================
    # PHASE 2: RAG/CoT processing for ALL patients
    # ==========================================================================
    logger.info(f"Phase 2: Running RAG processing for {len(patient_results)} patient(s)...")

    for patient_id, data in patient_results.items():
        token = set_request_id(patient_id)
        try:
            with log_timing(logger, "RAG processing"):
                with torch.no_grad():
                    run_rag_processing(
                        data["output_folder"],
                        data["top_trials_path"],
                        data["patient_info"],
                        model,
                        tokenizer,
                        config,
                    )

            with log_timing(logger, "Final ranking"):
                trial_data = load_trial_data(data["output_folder"])
                ranked_trials = rank_trials(trial_data)
                save_ranked_trials(
                    ranked_trials, str(Path(data["output_folder"]) / "ranked_trials.json")
                )

            logger.info("Pipeline completed for patient %s", patient_id)
        except Exception:  # noqa
            logger.exception("Phase 2 failed for patient %s", patient_id)
            continue
        finally:
            reset_request_id(token)


if __name__ == "__main__":
    main_pipeline()