import matplotlib.pyplot as pp
import data
import nnet

dataframe = data.read()
# pp.plot(dataframe['SP500'])

numpy_data = dataframe.values

# (training_data, test_data) = data.get_training_and_test_subsets(dataframe)

# data.fit_scaler_on(training_data)
# training_data = data.scale(training_data)

# (training_inputs, training_results) = data.get_inputs_and_results_of(training_data)

(
    training_inputs, training_results,
    test_inputs, test_results
    ) = data.get_completely_preprocessed_inputs_and_results_of(dataframe)

model = nnet.get_model()
nnet.compile(model)
nnet.train(model, training_inputs, training_results)

# print(model.evaluate(test_inputs, test_results))


# pp.ion()
# figure = pp.figure()
# subplot = figure.add_subplot(111)
# actual_line, = subplot.plot(test_results)
# predicted_line, = subplot.plot(test_results * .5)
# pp.show(block=True)

# prediction = model.predict(test_inputs, batch_size=len(test_inputs))
# predicted_line.set_ydata(prediction)

