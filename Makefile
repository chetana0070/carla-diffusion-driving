.PHONY: test validate phase0 phase1 phase2-validate phase2-audit phase3-prepare \
	phase4-model-smoke phase4-train-smoke phase4-train phase4-closed-loop \
	phase5-model-smoke phase5-train-smoke phase5-train phase5-latency \
	phase5-closed-loop phase5-expert-oracle phase5-diagnostics phase6-collect \
	phase6-audit phase6-prepare phase6-train-preflight phase6-residual-train \
	phase6-residual-closed-loop phase6-safety-smoke phase6-liveness-smoke \
	phase7-model-smoke phase7-train-smoke phase7-latency phase7-offline-gate \
	phase7-objective-smoke phase7-factorized-smoke phase7-factorized-train \
	phase7-factorized-latency phase7-factorized-gate phase7-longitudinal-smoke \
	phase7-longitudinal-train phase7-factorized-closed-loop \
	phase7-factorized-validate phase7-factorized-closed-loop-smoke \
	phase8-vla-prepare phase8-vla-validate phase8-vla-model-smoke \
	phase8-vla-train phase8-vla-sol-submit phase8-vla-latency \
	phase8-vla-offline-gate phase8-vla-corrective-smoke \
	phase8-vla-corrective-train phase8-vla-corrective-sol-submit \
	phase8-vla-corrective-latency phase8-vla-corrective-gate \
	phase8-vla-smoother-calibrate phase8-vla-smoother-latency \
	phase8-vla-smoother-gate phase8-vla-pareto-calibrate \
	phase8-vla-pareto-latency phase8-vla-pareto-development-gate

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

phase6-collect:
	./scripts/run_phase6_corrections.sh

phase6-audit:
	python scripts/audit_phase6_corrections.py

phase6-prepare:
	python scripts/prepare_phase6_training_data.py

phase6-train-preflight:
	python scripts/train_temporal_bc.py --processed-root data/processed/phase6_corrective_v2 --initial-checkpoint artifacts/checkpoints/phase5_temporal_bc_v080/best.pt --epochs 8 --learning-rate 0.00003 --freeze-encoder-epochs 8 --corrective-validation-split correction_validation --max-nominal-validation-degradation-fraction 0.05 --output-dir artifacts/checkpoints/phase6_dual_gate_v120

phase6-residual-train:
	python scripts/train_residual_correction.py

phase6-residual-closed-loop:
	./scripts/run_phase6_residual_closed_loop.sh

phase6-safety-smoke:
	CARLA_RENDER_MODE=live PHASE6_RESIDUAL_EPISODES=1 PHASE6_RESIDUAL_TICKS_PER_EPISODE=300 PHASE6_RESIDUAL_BACKGROUND_VEHICLES=2 PHASE5_SEED=20260901 PHASE6_RESIDUAL_REPORT=artifacts/evaluations/phase6_safety_smoke_v142.json PHASE6_RESIDUAL_TELEMETRY_DIR=artifacts/evaluations/phase6_safety_smoke_v142_telemetry PHASE6_RESIDUAL_SAVE_VIDEO=1 PHASE6_RESIDUAL_VIDEO_DIR=artifacts/evaluations/phase6_safety_smoke_v142_videos ./scripts/run_phase6_residual_closed_loop.sh
	python scripts/validate_phase6_safety_smoke.py --report artifacts/evaluations/phase6_safety_smoke_v142.json --telemetry-dir artifacts/evaluations/phase6_safety_smoke_v142_telemetry

phase6-liveness-smoke:
	CARLA_RENDER_MODE=live PHASE6_RESIDUAL_SEED=20260902 PHASE6_RESIDUAL_EPISODES=1 PHASE6_RESIDUAL_TICKS_PER_EPISODE=300 PHASE6_RESIDUAL_BACKGROUND_VEHICLES=8 PHASE6_RESIDUAL_REPORT=artifacts/evaluations/phase6_liveness_smoke_v153.json PHASE6_RESIDUAL_TELEMETRY_DIR=artifacts/evaluations/phase6_liveness_smoke_v153_telemetry PHASE6_RESIDUAL_SAVE_VIDEO=1 PHASE6_RESIDUAL_VIDEO_DIR=artifacts/evaluations/phase6_liveness_smoke_v153_videos ./scripts/run_phase6_residual_closed_loop.sh
	python scripts/validate_phase6_safety_smoke.py --report artifacts/evaluations/phase6_liveness_smoke_v153.json --telemetry-dir artifacts/evaluations/phase6_liveness_smoke_v153_telemetry

phase7-model-smoke:
	python scripts/smoke_phase7_diffusion_model.py

phase7-train-smoke:
	python scripts/train_diffusion_policy.py --smoke --output-dir artifacts/checkpoints/phase7_diffusion_smoke

phase7-latency:
	python scripts/benchmark_diffusion_latency.py --checkpoint artifacts/checkpoints/phase7_diffusion_smoke/best.pt

phase7-offline-gate:
	python scripts/evaluate_diffusion_offline.py

phase7-objective-smoke:
	python scripts/train_diffusion_policy.py --smoke --initial-checkpoint \
		artifacts/checkpoints/phase7_diffusion_v161/best.pt \
		--output-dir artifacts/checkpoints/phase7_objective_smoke_v180

phase7-factorized-smoke:
	python scripts/train_factorized_policy.py --smoke --allow-cpu \
		--output-dir artifacts/checkpoints/phase7_factorized_smoke_v190

phase7-factorized-train:
	python scripts/train_factorized_policy.py \
		--initial-diffusion-checkpoint \
		artifacts/checkpoints/phase7_diffusion_corrected_v180/best.pt \
		--output-dir artifacts/checkpoints/phase7_factorized_v190

phase7-factorized-latency:
	python scripts/benchmark_factorized_policy_latency.py \
		--checkpoint artifacts/checkpoints/phase7_factorized_v190/best.pt

phase7-factorized-gate:
	python scripts/evaluate_factorized_offline.py \
		--factorized-checkpoint artifacts/checkpoints/phase7_factorized_v190/best.pt \
		--latency-report artifacts/evaluations/phase7_factorized_latency_v190.json \
		--output artifacts/evaluations/phase7_factorized_offline_gate_v190.json

phase7-longitudinal-smoke:
	python scripts/finetune_factorized_longitudinal.py --smoke --allow-cpu \
		--initial-checkpoint artifacts/checkpoints/phase7_factorized_v191/best.pt \
		--output-dir artifacts/checkpoints/phase7_longitudinal_smoke_v200

phase7-longitudinal-train:
	python scripts/finetune_factorized_longitudinal.py \
		--initial-checkpoint artifacts/checkpoints/phase7_factorized_v191/best.pt \
		--output-dir artifacts/checkpoints/phase7_factorized_longitudinal_v200

phase7-factorized-closed-loop:
	./scripts/run_phase7_factorized_closed_loop.sh

phase7-factorized-validate:
	python scripts/validate_phase7_factorized_closed_loop.py

phase7-factorized-closed-loop-smoke:
	CARLA_RENDER_MODE=live PHASE7_SEED=20260901 PHASE7_EPISODES=1 \
		PHASE7_TICKS_PER_EPISODE=100 PHASE7_BACKGROUND_VEHICLES=2 \
		PHASE7_REPORT=artifacts/evaluations/phase7_factorized_smoke_v210.json \
		PHASE7_TELEMETRY_DIR=artifacts/evaluations/phase7_factorized_smoke_v210_telemetry \
		PHASE7_SAVE_VIDEO=1 \
		PHASE7_VIDEO_DIR=artifacts/evaluations/phase7_factorized_smoke_v210_videos \
		./scripts/run_phase7_factorized_closed_loop.sh

phase8-vla-prepare:
	python scripts/prepare_phase8_vla_data.py

phase8-vla-validate:
	python scripts/validate_phase8_vla_data.py \
		--report artifacts/evaluations/phase8_vla_data_v220.json

phase8-vla-model-smoke:
	python scripts/smoke_phase8_vla_model.py

phase8-vla-train:
	python scripts/train_phase8_vla.py --pretrained-visual-encoder

phase8-vla-sol-submit:
	sbatch scripts/slurm/train_phase8_vla.sbatch

phase8-vla-latency:
	python scripts/benchmark_phase8_vla_latency.py \
		--checkpoint artifacts/checkpoints/phase8_vla_v220/best.pt \
		> artifacts/evaluations/phase8_vla_latency_v230.json

phase8-vla-offline-gate:
	python scripts/evaluate_phase8_vla_offline.py \
		--checkpoint artifacts/checkpoints/phase8_vla_v220/best.pt \
		--training-report artifacts/checkpoints/phase8_vla_v220/report.json \
		--latency-report artifacts/evaluations/phase8_vla_latency_v230.json \
		--output artifacts/evaluations/phase8_vla_offline_gate_v230.json

phase8-vla-corrective-smoke:
	python scripts/finetune_phase8_vla.py \
		--smoke --allow-cpu --workers 0 \
		--initial-checkpoint artifacts/checkpoints/phase8_vla_v220/best.pt \
		--output-dir artifacts/checkpoints/phase8_vla_corrective_smoke_v240

phase8-vla-corrective-train:
	python scripts/finetune_phase8_vla.py \
		--initial-checkpoint artifacts/checkpoints/phase8_vla_v220/best.pt \
		--output-dir artifacts/checkpoints/phase8_vla_corrective_v240

phase8-vla-corrective-sol-submit:
	sbatch scripts/slurm/finetune_phase8_vla.sbatch

phase8-vla-corrective-latency:
	python scripts/benchmark_phase8_vla_latency.py \
		--checkpoint artifacts/checkpoints/phase8_vla_corrective_v240/best.pt \
		> artifacts/evaluations/phase8_vla_corrective_latency_v240.json

phase8-vla-corrective-gate:
	python scripts/evaluate_phase8_vla_offline.py \
		--checkpoint artifacts/checkpoints/phase8_vla_corrective_v240/best.pt \
		--training-report artifacts/checkpoints/phase8_vla_corrective_v240/report.json \
		--latency-report artifacts/evaluations/phase8_vla_corrective_latency_v240.json \
		--output artifacts/evaluations/phase8_vla_corrective_gate_v240.json

phase8-vla-smoother-calibrate:
	python scripts/calibrate_phase8_vla_smoother.py \
		--base-checkpoint artifacts/checkpoints/phase8_vla_corrective_v240/best.pt \
		--base-training-report artifacts/checkpoints/phase8_vla_corrective_v240/report.json \
		--output-dir artifacts/checkpoints/phase8_vla_smoothed_v250

phase8-vla-smoother-latency:
	python scripts/benchmark_phase8_vla_latency.py \
		--checkpoint artifacts/checkpoints/phase8_vla_smoothed_v250/best.pt \
		> artifacts/evaluations/phase8_vla_smoothed_latency_v250.json

phase8-vla-smoother-gate:
	python scripts/evaluate_phase8_vla_offline.py \
		--checkpoint artifacts/checkpoints/phase8_vla_smoothed_v250/best.pt \
		--training-report artifacts/checkpoints/phase8_vla_smoothed_v250/report.json \
		--latency-report artifacts/evaluations/phase8_vla_smoothed_latency_v250.json \
		--output artifacts/evaluations/phase8_vla_smoothed_gate_v250.json

phase8-vla-pareto-calibrate:
	python scripts/calibrate_phase8_vla_smoother.py \
		--selection-strategy pareto_robust \
		--base-checkpoint artifacts/checkpoints/phase8_vla_corrective_v240/best.pt \
		--base-training-report artifacts/checkpoints/phase8_vla_corrective_v240/report.json \
		--prior-gate-report artifacts/evaluations/phase8_vla_smoothed_gate_v250.json \
		--output-dir artifacts/checkpoints/phase8_vla_pareto_v260

phase8-vla-pareto-latency:
	python scripts/benchmark_phase8_vla_latency.py \
		--checkpoint artifacts/checkpoints/phase8_vla_pareto_v260/best.pt \
		> artifacts/evaluations/phase8_vla_pareto_latency_v260.json

phase8-vla-pareto-development-gate:
	python scripts/evaluate_phase8_vla_offline.py \
		--checkpoint artifacts/checkpoints/phase8_vla_pareto_v260/best.pt \
		--training-report artifacts/checkpoints/phase8_vla_pareto_v260/report.json \
		--latency-report artifacts/evaluations/phase8_vla_pareto_latency_v260.json \
		--output artifacts/evaluations/phase8_vla_pareto_development_gate_v260.json
