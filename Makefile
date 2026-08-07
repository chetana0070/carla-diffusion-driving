.PHONY: test validate phase0 phase1 phase2-validate phase2-audit

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
