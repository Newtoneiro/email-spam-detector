# Email Spam Detector

The project, titled "Email Spam Detector," focuses on creating a system to classify emails as either spam or ham (non-spam). The project is divided into three main parts: data processing, model training, and a user-friendly interface. The selected dataset undergoes preprocessing and analysis, followed by training and comparison of various machine learning models. The Random Forest Classifier is identified as the top-performing model.

## Project Structure

The project is organized into the following main parts:

1. **Data:**

   - `data/dataset_selection.ipynb`: Notebook for selecting the dataset for the project.
   - `data/data_analysis_and_preprocess.ipynb`: Notebook for dataset preprocessing and analysis.

2. **Model Training and Comparison:**

   - `models/models_comparison.ipynb`: Folder containing Python scripts for training and evaluating different models.

3. **User Interface:**

   - `main_app.py`: Folder containing the main application script.

4. **Reports:**

   - `report.pdf`: Summary report of the project.

## How to Run

Follow these steps to run the Email Spam Detector:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Newtoneiro/email-spam-detector
   cd email-spam-detector

   ```

2. **Install Dependencies:**

   ```
   pip install -r requirements.txt

   ```

3. **Run the User Interface:**

   ```
   python3 -m streamlit run src/main_app.py

   ```

4. **Use the app:**
   - Input your email content in the provided text box.
   - Click the "Predict" button to receive the prediction of whether the email is spam or ham.

_Note: Ensure you have Python and pip installed on your system before running the application._
