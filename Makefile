help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

args ?= -vvv --cov multiscalemnist
test: ## Run tests
	pytest $(args)

shell: ## Run poetry shell
	poetry shell
