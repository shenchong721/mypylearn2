Pylearn2 in practice
====================

See [http://fastml.com/pylearn2-in-practice/](http://fastml.com/pylearn2-in-practice/) for description.

Here are the commands to execute for Windows:

	set PYTHONPATH=%PYTHONPATH%;.
	set THEANO_FLAGS=device=cpu
	train.py adult.yaml
	python predict.py softmax_regression_best.pkl adult\test.csv predictions.txt
	
And for Unix (untested):

	export PYTHONPATH=$PYTHONPATH:.
	export THEANO_FLAGS=device=cpu
	train.py adult.yaml
	python predict.py softmax_regression_best.pkl adult/test.csv predictions.txt
	or :
	python mypredict.py softmax_regression_best.pkl adult/test.csv predictions.txt

url :
	git code : https://github.com/zygmuntz/pylearn2-practice
	http url : http://fastml.com/pylearn2-in-practice/

features :
	how to predict test with trained  pkl file