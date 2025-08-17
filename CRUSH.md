# CRUSH.md

## Project Setup
- **Dependencies**: Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```

## Commands
- **Build**: No explicit build process except dependency installation.
- **Lint**: Use `flake8` or `pylint` for linting Python code.
- **Test**: Consider using `pytest` to run tests once they're available.

## Code Style
- **Imports**: Use absolute imports, example:
  ```python
  from langchain_core.prompts import PromptTemplate
  ```
- **Formatting**: Follow PEP 8 guidelines:
  - 4 spaces per indentation level
  - Lines to a maximum of 79 characters
  - Two blank lines before top-level functions and classes
- **Types**: Use type annotations, example:
  ```python
  def example_function(param: Type) -> ReturnType:
  ```
- **Naming Conventions**: 
  - Functions: `lower_case_with_underscores`
  - Classes: `CamelCase`

## Error Handling
- Use structured error handling with exceptions.

## Best Practices
- Add `.crush` directory to `.gitignore` if necessary.

