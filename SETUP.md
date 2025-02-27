## 🛠️ Environment Setup

This implementation using Python `3.9.18`. I recommend using `pyenv` to install the python version.

### Using `pip` and `venv` 📜

1. Navigate to the project root:
    ```bash
    cd path/to/transformer-from-scratch
    ```
2. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```

3. Activate the environment:

- MacOS/Linux: `source .venv/bin/activate`
- Windows: `.\.venv\Scripts\activate`

4. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Verify installation:

    ```bash
    pip list
    ```

### Or using Conda 🐍

1. Navigate to the project root:

    ```bash
    cd path/to/transformer-from-scratch
    ```

2. Create a conda environment:

    ```bash
    conda create --name <env_name> python=3.9.18
    ```

3. Activate the environment:

    ```bash
    conda activate <env_name>
    ```

4. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Verify installation:

    ```bash
    conda list
    ```

