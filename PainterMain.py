from tkinter  import *
from MnistClasifier import *
from PIL import ImageGrab
import torch
class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'white'

    def __init__(self):
        self.root = Tk()

        self.clasify_button = Button(self.root, text='Clasify', command = self.classify)
        self.clasify_button.grid(row=0, column=1)
        self.clasify_button = Button(self.root, text='Reset', command = self.clear)
        self.clasify_button.grid(row=0, column=2)
        self.text = Label(self.root, text="")
        self.text.grid(row=0, column=3)
        self.c = Canvas(self.root, bg='black', width=320, height=320)
        self.c.grid(row=1, columnspan=5)
        self.conf = Label(self.root, text="")
        self.conf.grid(row=2, columnspan=5, rowspan=4)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.draw_confidence(torch.zeros([1, 10], dtype=torch.int32))

    def getter(self, widget, path):
        x=self.root.winfo_rootx()+widget.winfo_x()
        y=self.root.winfo_rooty()+widget.winfo_y()
        x1=x+widget.winfo_width()
        y1=y+widget.winfo_height()
        ImageGrab.grab().crop((x+2,y+2,x1-2,y1-2)).save(path)

    def classify(self):
        path = "test.jpg"
        self.getter(self.c, path)
        result = recognize(path)
        self.text['text'] = result[0]
        self.draw_confidence(result[1])

    def draw_confidence(self, confidence_tensor):
        out = "0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n"
        print(type(confidence_tensor))
        for i in range(10):
            
            out = out+str(int(confidence_tensor[0][i].item() * 100))+"%\t"
        self.conf['text'] = out

    def clear(self):
        print("reset")
        self.c.delete("all")
        self.draw_confidence(torch.zeros([1, 10], dtype=torch.int32))


    def activate_button(self, some_button, eraser_mode=False):
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = 15
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=True, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    ge = Paint()