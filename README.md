# DeepCraft-Intern

## Stock Price Prediction using LSTM

This Streamlit application predicts stock prices using **LSTM** (Long Short-Term Memory) neural networks. The app allows users to interactively **adjust** key model parameters and **visualize** both actual and predicted stock prices, providing insights into stock market trends and model performance.

### Features:
- **Interactive Sliders**: Customize the number of days (`n_input`) for forecasting and set the number of neurons in each LSTM layer (three layers in total).
- **Data Visualization**: Graphs to visualize stock prices over time, along with training and test predictions, and a future stock price forecast.
- **Model Performance Metrics**: RMSE (Root Mean Squared Error) calculation to evaluate model performance on both training and testing datasets.

---

## Installation

### Step 1: Clone the repository
To get started, clone this repository:
```bash
git clone https://github.com/BhanuPratap16/DeepCraft-Intern.git
```

### Step 2 : Change Directory
Move into the ==project== directory:

```bash
cd DeepCraft-Intern
```


### Step 3 : Create Virtual environment 
Create and activate a virtual environment:
``` bash
# For Linux/macOS
python3 -m venv env
source env/bin/activate

# For Windows
python -m venv env
.\env\Scripts\activate

```



### Step 4: Install dependencies
Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 5: Run the application
Start the `Streamlit` app by running the following command:

```bash
streamlit run app.py
```

This will launch the Streamlit application in your `web browser`.