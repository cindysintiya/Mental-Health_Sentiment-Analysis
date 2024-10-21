from website import create_app
from data.model import SentimentAnalysisModel

if __name__ == "__main__" :
    app = create_app()
    app.run(debug=True)   # auto reload jk file berubah, tdk bs krn auto ada pycache        
    # app.run(debug=False)