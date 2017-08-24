import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#-------------------LOAD DATA

#Load training data
training_data_df = pd.read_csv("data/sales_data_training.csv", dtype=float)

#Decompose X (training data) and Y (prediction) from training data
X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[['total_earnings']].values

#Load testing data set
test_data_df = pd.read_csv('data/sales_data_test.csv',dtype=float)

#Decompose X (training data) and Y (prediction) from training data
X_testing = test_data_df.drop('total_earnings',axis=1).values
Y_testing = test_data_df[['total_earnings']].values

#Data needs to be normalized, lets define the appropiate ranges
X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1))

#Scale the training and test data sets
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)
print("Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0],Y_scaler.min_[0]))

#------------------- DESCRIPTION OF MODEL
#Model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5
#define the number of inputs and outputs
number_of_inputs = 9
number_of_outputs = 1

#Define the number of neurons in each layer of the neural net
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

#------------------- DEFINITIONS OF NEURAL NETWORK LAYERS
#Input layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32,shape=(None,number_of_inputs))

#Layer 1
with tf.variable_scope('layer_1'):
    #weights connects each node with a node in the previous layer
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    #bias value for each node
    biases =  tf.get_variable(name="biases1", shape=[layer_1_nodes],initializer = tf.zeros_initializer())
    #activation function that outputs the result of the layer
    layer_1_output = tf.nn.relu(tf.matmul(X,weights) + biases)

#Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases =  tf.get_variable(name="biases2", shape=[layer_2_nodes],initializer = tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

#Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases =  tf.get_variable(name="biases3", shape=[layer_3_nodes],initializer = tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

#Output layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases =  tf.get_variable(name="biases4", shape=[number_of_outputs],initializer = tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

#Define a cost function
with tf.variable_scope('cost'):
    #Y is the expected value that will be feed under training
    Y = tf.placeholder(tf.float32, shape =(None,1))
    cost = tf.reduce_mean(tf.squared_difference(prediction,Y))

#Define an optimization method
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#This will create a summary operation to log  the progress of the training
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost',cost)
    summary = tf.summary.merge_all()

#------------------- SAVER
saver = tf.train.Saver()
#------------------- TRAINING LOOP
with tf.Session() as session:

    #Start the session by initializen all variables and layers
    session.run(tf.global_variables_initializer())

    #Record the training progress
    training_writer = tf.summary.FileWriter("./logs/training", session.graph)
    testing_writer = tf.summary.FileWriter("./logs/testing", session.graph)

    #Now define the training LOOP
    for epoch in range(training_epochs):

        #Training operation: The process consist in feeding the neural net to the optimizer function
        session.run(optimizer, feed_dict={X:X_scaled_training, Y:Y_scaled_training})

        if epoch % 5 == 0:
            training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_scaled_training, Y:Y_scaled_training})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})
            print(epoch, training_cost, testing_cost)

            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

    print("Training Complete")

    #Print the final training and testing costs
    final_training_cost = session.run(cost, feed_dict={X:X_scaled_training, Y:Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X:X_scaled_testing, Y:Y_scaled_testing})
    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))

    #Lets run the prediction operation
    Y_predicted_scaled = session.run(prediction,  feed_dict={X:X_scaled_testing})
    #Unscale the data back to its original units
    Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

    #Y_predicted has now a prediction obtained from our trained model,
    #lets compare the results with the actual earning values
    real_earnings = test_data_df['total_earnings'].values[0]
    predicted_earnings = Y_predicted[0][0]

    print("The actual earnings of Game #1 were ${}".format(real_earnings))
    print("Our neural network predicted earnings of ${}".format(predicted_earnings))
#------------------- SAVING FOR CHECKPOINTS
    save_path = saver.save(session, "logs/trained_model.ckpt")
    print("Model saved:{}".format(save_path))

#------------------- SAVING FOR DEPLOYMENT
    model_builder = tf.saved_model.builder.SavedModelBuilder("exported_model")
    inputs = {
        'input': tf.saved_model.utils.build_tensor_info(X)
    }
    outputs = {
        'earnings': tf.saved_model.utils.build_tensor_info(prediction)
    }
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs = inputs,
        outputs = outputs,
        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    model_builder.add_meta_graph_and_variables(
        session,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        }
    )

    model_builder.save()
