.PHONY: test validate phase0 phase1 phase2-validate phase2-audit phase3-prepare phase4-model-smoke phase4-train-smoke phase4-train phase4-closed-loop phase5-model-smoke phase5-train-smoke phase5-train phase5-latency phase5-closed-loop phase5-expert-oracle phase5-diagnostics

test:
	python -m unittest discover -s tests -v

validate:
	python scripts/validate_config.py
	python scripts/validate_system.py --json-out artifacts/system_report.json

phase0: test validate

phase1:
	./scripts/run_phase1_smoke.sh

phase2-validate:
	python scripts/validate_dataset.py data/raw/phase2_pilot

phase2-audit:
	python scripts/audit_phase2_dataset.py data/raw/phase2_pilot

phase3-prepare:
	python scripts/prepare_phase3_data.py data/raw/phase2_pilot_v2_v043

phase4-model-smoke:
	python scripts/smoke_phase4_model.py

phase4-train-smoke:
	python scripts/train_single_frame_bc.py --smoke --output-dir artifacts/checkpoints/phase4_single_frame_smoke

phase4-train:
	python scripts/train_single_frame_bc.py --pretrained

phase4-closed-loop:
	./scripts/run_phase4_closed_loop.sh

phase5-model-smoke:
	python scripts/smoke_phase5_temporal_model.py

phase5-train-smoke:
	python scripts/train_temporal_bc.py --smoke --output-dir artifacts/checkpoints/phase5_temporal_smoke

phase5-train:
	python scripts/train_temporal_bc.py --pretrained

phase5-latency:
	python scripts/benchmark_temporal_bc_latency.py

phase5-closed-loop:
	./scripts/run_phase5_closed_loop.sh

phase5-expert-oracle:
	./scripts/run_phase5_expert_oracle.sh

phase5-diagnostics:
	python scripts/summarize_phase5_diagnostics.py
