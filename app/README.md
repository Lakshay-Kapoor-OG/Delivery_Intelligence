# Delivery Operations Intelligence System

## Requirements
- Python 3.10 or newer
- pip

## Setup on Windows

1. Copy the complete `Delivery_Intelligence` project folder to your computer.
2. Open Command Prompt inside the `app` folder.
3. Create a virtual environment:

   python -m venv .venv

4. Activate it:

   .venv\Scripts\activate

5. Install dependencies:

   python -m pip install --upgrade pip
   pip install -r requirements.txt

6. Launch the dashboard:

   streamlit run app.py

7. Open the local URL shown in the terminal, usually:
   http://localhost:8501

## Notes
- Keep the `data` and `models` folders beside `app.py`.
- The model predicts delivery duration from the supplied order characteristics.
- Predictions are estimates, not guaranteed delivery times.
- This is a prototype built using a public dataset. Its results may not generalize to live operations.
