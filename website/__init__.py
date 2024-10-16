from flask import Flask, render_template

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = "secret!hahaha"  # kunci keamanan

    # import semua route
    from .loading import loading
    from .analysis import analysis
    
    # buat route
    app.register_blueprint(loading)
    app.register_blueprint(analysis)

    @app.errorhandler(404)   # url/ halaman tdk ditemukan
    def not_found(e) :
        return render_template("not found.html", title='Page Not Found', content='404 Page Not Found')

    @app.errorhandler(405)   # url tdk sesuai method yg tersimpan
    def not_allowed(e) :
        return render_template("not found.html", title='URL Not Allowed', content='You are not allowed to access this URL')

    @app.errorhandler(500)   # kesalahan/ error di server
    def internal_error(e) :
        return render_template("not found.html", title='Server Error', content='Something went wrong. Sorry :(')

    return app