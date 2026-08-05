.PHONY: test validate phase0

test:
	python -m unittest discover -s tests -v

validate:
	python scripts/validate_config.py
	python scripts/validate_system.py --json-out artifacts/system_report.json

phase0: test validate

