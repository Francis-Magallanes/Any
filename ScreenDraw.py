import turtle
from turtle import Turtle, Screen
from tkinter import messagebox
import pygetwindow
import pyautogui
from PIL import Image
import NeuralNetworkScratch as nns
import cv2
import numpy as np
import matplotlib.pyplot as plt

fileName = "Any"
windowname = "Any - 0 to 9 Digit Recognition"

#his will build the screen
screen = Screen()
screen.setup(width = 350, height=350, startx = None, starty=None)
t = Turtle("turtle")
t.pen(pensize = 36)
t.speed(-1)

#Initializing the trained network
ann = nns.NeuralNetwork()

def predictdrawing():

    image_orig =  cv2.imread(fileName+".png")
    image_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    image_gray_resized = cv2.resize(image_gray,(28,28))
    #cv2.imwrite("adjusted_img.png",image_gray_resized)
    #plt.imshow(image_gray_resized, cmap = "Greys")

    #adjustment for the feeding to the neural network
    seperator =  112.5
   # print(image_gray_resized)
    image_gray_resized_adj = np.where(image_gray_resized < seperator, 0.99, image_gray_resized)
    image_gray_resized_adj =  np.where(image_gray_resized_adj >= seperator, 0.01, image_gray_resized_adj)

    plt.imshow(image_gray_resized_adj, cmap = "Greys")
    plt.show()

    image_gray_resized_adj = np.reshape(image_gray_resized_adj, (784,1))
    #cv2.imshow("Test", image_gray_resized_adj.reshape(28,28))

    #prediction proper
    results = ann.predict(image_gray_resized_adj)

    print(results)
    get_result = np.argmax(results)

    return get_result

def dragging(x, y):
    t.ondrag(None)
    t.setheading(t.towards(x, y))
    t.goto(x, y)
    t.ondrag(dragging)

def leftclick(x,y):
    t.penup()
    t.goto(x, y)
    t.pendown()

def screenclear(x, y):
    t.clear()

def SaveImage():
    screen.tracer(False)
    screen.tracer(True)
    canvas = screen.getcanvas()
    canvas.postscript(file= fileName+'.eps',width=28, height=28)

    img = Image.open(fileName+'.eps')
    img.save(fileName+'.jpg')

def WindowScreenhot():
    t.hideturtle()

    x1_offset = 15
    y1_offset = 40
    x2_offset = -40
    y2_offset = -40

    window = pygetwindow.getWindowsWithTitle(windowname)[0]
    x1,y1,width,height =window.left, window.top, window.width, window.height
    x2,y2 = x1 + width, y1 + height

    window_dim = (x1+x1_offset, y1+y1_offset, x2 + x2_offset, y2 + y2_offset) 

    print(window_dim)

    pyautogui.screenshot(fileName+".png")
    img = Image.open(fileName+".png")
    img = img.crop(window_dim)
    img.save(fileName+".png")

    pred = predictdrawing()
    messagebox.showinfo("Turtle say its....",  f"Turtle say its a number {pred}")
    t.showturtle()



def main():

    #event listener
    turtle.listen()
    t.ondrag(dragging)
    turtle.onscreenclick(leftclick, 1) #event when left click is clicked
    turtle.onscreenclick(screenclear, 3) #event when right click is clicked
    turtle.onkey(WindowScreenhot, 'space')
    
    screen.title(windowname)
    screen.mainloop()

    


    
main()