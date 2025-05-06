# ModeMetrics
Name: Mariam Thabti  
Student ID: 210328471  
QMUL Final Year Project  
Git Repository: `https://github.com/mariamth/ModeMetrics.git`


There is no executable file for this code, it needs to be built and ran. As this project is not deployed, DEBUG == True for ease of marking and presenting this task.

This is a Django and React webapp that fetches data from ASOS via rapidapi.com. It then stores the data in django backend where a model is trained on it to classify how trendy it is. It is paired with a frontend search functionality, which displays a verdict and adivce.

Python version 3.11.11

# Build instructions

The .env file with my API key is included for the purpose of marking this project. 
---
## Backend (Django) setup
Navigate to the backend directory:

```
cd backend
```

Set up virtual environment either using venv 

```
python -m venv venv
source venv/bin/activate     
#if using windows: venv\Scripts\activate
pip install -r requirements.txt
```
or conda

```
conda create --name modemetrics python=3.11
conda activate modemetrics
pip install -r requirements.txt
```

Next, apply migrations and runserver:

```
python manage.py migrate
python manage.py runserver
```
This is for the Django backend and will fetch the data from the API, store it, train the model, predict the trendiness and allow for the search similarity logic to occur. 
---

## Frontend Setup

In a new terminal:

```
cd modemetrics
npm install
npm install axios web-vitals
npm start
```

The frontend will run at `http://localhost:3000`
This is the main location for the web application.

The backend runs at `http://127.0.0.1:8000` see views for future possiblities



---

## How to use the site

1. Enter shoe related keywords (e.g. "Loafers")
2. The app sends a query to /api/similar/?q=...
3. This compares the keyword to product names stored
4. Displays a verdict based on model predictions

---

## Testing
To run the precreated testing which runs through 3 tests: a trending style, a trending style and when no similar products are available
For this test to work, the django server and react page must be running, which will be the case if the steps above are followed. Make sure your environment is also running.
This test only works on Chrome.
### Django tests:

In the same backend directory as when running the django server:

```
python manage.py test
```

---
Django admin panel was not needed for this project and therefore was not included.
If you would like to check, create a superuser and follow the prompts:

```
python manage.py createsuperuser
```