from flask import Flask,render_template
app = Flask(__name__,static_url_path='/static')



@app.route("/")
def hello():
   return render_template("index.html")

@app.route("/about")
def about():
   return render_template("about.html")

@app.route("/contact")
def contact():
   return render_template("contact.html")


@app.route("/delivery")
def delivery():
   return render_template("delivery.html")

@app.route("/news")
def news():
   return render_template("news.html")

@app.route("/preview")
def preview():
   return render_template("preview.html")

@app.route("/receivedata", methods=['POST'])
def receive_data():
    print (request.form["myData"])

if __name__ == '__main__':
    app.run(debug=True,port=3000)