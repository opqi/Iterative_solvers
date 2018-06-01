import os
import numpy as np
import matplotlib.pyplot as pyplot


DIRS = ["./results/add32",
		"./results/jpwh_991"
		]

COLORS = ["b-<","r-H","g-*","k-^","c-|","b->","y-_"]

X = np.logspace(-1,-15,num=35)

def get_plots(dir):
	plots_acc = []
	plots_time = []
	for root,dirs,files in os.walk(dir):
		for file in files:
			if file.endswith(".acc"):
				with open(os.path.join(root,file), "r") as f:
					arr = f.readlines()
					plots_acc.append((file[:-4],list(map(float,arr))))
			if file.endswith(".time"):
				with open(os.path.join(root,file), "r") as f:
					arr = f.readlines()
					plots_time.append((file[:-5],list(map(float,arr))))
	return plots_acc, plots_time

for d in DIRS:
	dirname = os.path.basename(d)
	
	plots_acc, plots_time = get_plots(d)
	pyplot.figure()
	for c, (name, Y) in zip(COLORS,plots_acc):
		pyplot.loglog(X,Y,c,label=name)
		#pyplot.plot(np.log10(X),np.log(Y),c,label=name)
	pyplot.legend(loc="best")
	pyplot.title("Matrix \"{}\" Accuracy".format(dirname))
	pyplot.xlabel("Tolerance")
	pyplot.ylabel("Accuracy")
	pyplot.gca().invert_xaxis()

	pyplot.figure()
	for c, (name, Y) in zip(COLORS,plots_time):
		pyplot.semilogx(X,Y,c,label=name)
		#pyplot.plot(np.log10(X),np.log(Y),c,label=name)
	pyplot.legend(loc="best")
	pyplot.title("Matrix \"{}\" Time".format(dirname))
	pyplot.xlabel("Tolerance")
	pyplot.ylabel("Elapsed time, sec")
	pyplot.gca().invert_xaxis()

pyplot.show()