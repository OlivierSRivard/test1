Get-ChildItem -Recurse -File -Include Home.py,home.py,app.py,main.py,streamlit_app.py `
| Select-Object FullName
