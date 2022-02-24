IN = ./src/
OUT = ./obj/

all: MnistTraining.exe MnistClassification.exe

MnistTraining.exe: MnistTraining.o NeuralNetwork.o Algebra.o
	gcc -o MnistTraining.exe $(OUT)MnistTraining.o $(OUT)NeuralNetwork.o $(OUT)Algebra.o -lm
	mv *.exe $(OUT)

MnistClassification.exe: MnistClassification.o NeuralNetwork.o Algebra.o
	gcc -o MnistClassification.exe $(OUT)MnistClassification.o $(OUT)NeuralNetwork.o $(OUT)Algebra.o -lm
	mv *.exe $(OUT)

MnistTraining.o: $(IN)MnistTraining.c $(IN)NeuralNetwork.h
	gcc -c $(IN)MnistTraining.c
	mv *.o $(OUT)

MnistClassification.o: $(IN)MnistClassification.c $(IN)NeuralNetwork.h
	gcc -c $(IN)MnistClassification.c
	mv *.o $(OUT)

NeuralNetwork.o: $(IN)NeuralNetwork.c $(IN)NeuralNetwork.h $(IN)Algebra.h
	gcc -c $(IN)NeuralNetwork.c -lm
	mv *.o $(OUT)

Algebra.o: $(IN)Algebra.c $(IN)Algebra.h
	gcc -c $(IN)Algebra.c -lm
	mv *.o $(OUT)

clean:
	rm -f $(OUT)*.o
	rm -f $(OUT)*.exe