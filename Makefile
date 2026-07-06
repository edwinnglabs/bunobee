.PHONY: version bump bump-dev

version:  ## Print the current version
	@python scripts/bump_version.py --show

bump:  ## Bump the patch version within the 0.0.x lane (0.0.4 -> 0.0.5)
	@python scripts/bump_version.py

bump-dev:  ## Bump/append a .devN pre-release (0.0.4 -> 0.0.5.dev0)
	@python scripts/bump_version.py --dev
