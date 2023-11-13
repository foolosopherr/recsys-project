# CI/CD Workflow

This repository is configured with a Continuous Integration/Continuous Deployment (CI/CD) workflow using GitHub Actions. The workflow is triggered on each push to the main branch and runs a series of checks and tasks to ensure code quality and reliability.

## Workflow Steps

1. **Lint code base:**
   - Uses Flake8 to lint Python code.
   - Runs Pytest for unit tests.
   - Runs MyPy for static type checking.

2. **Format code with black:**
   - Uses Black to automatically format the code according to a consistent style.

3. **Check for changes after formatting:**
   - Verifies that the code changes made by Black are accepted.

## Usage

- On every push to the main branch, the CI/CD workflow is triggered automatically.
- The workflow checks for linting errors, runs tests, performs code formatting, and ensures that no changes are left uncommitted after formatting.

## Additional Notes

- The workflow is configured to run on Ubuntu latest.
- It uses Python 3.8 for testing and linting tasks.

Feel free to customize this workflow to suit your project's specific needs.

**Note:** If the workflow fails due to formatting changes, you might need to review and commit the changes manually.

**Dataset source** https://www.kaggle.com/datasets/sharthz23/mts-library
