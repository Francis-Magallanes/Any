import turtle
from turtle import Turtle, Screen
from PIL import Image

fileName = "Any"
screen = Screen()
screen.setup(28, 28)
t = Turtle("turtle")
t.speed(-1)

def dragging(x, y):
    t.ondrag(None)
    t.setheading(t.towards(x, y))
    t.goto(x, y)
    t.ondrag(dragging)

def screenclear(x, y):
    t.clear()

def SaveImage():
    screen.tracer(False)
    screen.tracer(True)
    canvas = screen.getcanvas()
    canvas.postscript(file= fileName+'.eps',width=28, height=28)

    img = Image.open(fileName+'.eps')
    img.save(fileName+'.jpg')

def main():
    turtle.listen()

    t.ondrag(dragging)

    turtle.onscreenclick(screenclear, 3)

    turtle.onkey(SaveImage, 'space')

    screen.mainloop()

main()