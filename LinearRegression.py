""" a linear regression model practice with two weights and one bias
set some strange noise and sample distribution to try to find more.
But when the learning rate is larger than 0.0002, gradient explose,
learning failed.
Just ignore the first and most basic model.
Then try to find more about tensorflow in my following codes."""



import tensorflow as tf


SAMPLE_SIZE = 100000
EPOCH_NUM = 30000

LR = 0.0002


def init_data():
    x1 = tf.constant(tf.random.uniform([SAMPLE_SIZE,1],-100.0,100.0,dtype=tf.float32))
    x2 = tf.constant(tf.random.normal([SAMPLE_SIZE,1],mean=4.5,stddev=61.7,dtype=tf.float32))
    y = tf.constant(0.6*x1+3.9*x2+tf.random.normal([SAMPLE_SIZE,1],mean=0,stddev=0.2,dtype=tf.float32)+
                    tf.constant(0.9,shape=[SAMPLE_SIZE,1]),dtype=tf.float32)
    return tf.concat([x1,x2],-1),y

class LinearModel(tf.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.weights = tf.Variable(tf.random.uniform([2,1],-5,5),name='weights',dtype=tf.float32)
        self.bias = tf.Variable(0.0)

    def __call__(self,x):
        return tf.matmul(x,self.weights) + self.bias

def loss(tar_y,pre_y):
    return tf.reduce_mean(tf.square(tar_y-pre_y))

@tf.function
def epoch_train(model,x,y):
    with tf.GradientTape() as tp:
        current_loss = loss(y,model(x))

    dw,db = tp.gradient(current_loss,[model.weights,model.bias])
    model.weights.assign_sub(LR*dw)
    model.bias.assign_sub(LR*db)


model = LinearModel()
x1_x2,y = init_data()

def training_model(model,x,y):
    for epoch in range(EPOCH_NUM):
        epoch_train(model,x,y)
        print("Current weights are: ",model.weights.numpy())
        print("Current bias is: ",model.bias.numpy())
        print("Current Loss is: ",loss(y,model(x)).numpy())
        print("-----------------------EPOCH{}-----------------".format(epoch))

training_model(model,x1_x2,y)
