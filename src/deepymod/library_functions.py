import numpy as np
import tensorflow as tf

'''
This file contains several ready-to-use library functions.
'''

def library_1D(data, prediction, library_config):
    '''
    Constructs a library graph in 1D. Library config is dictionary with required terms.
    '''

    # Polynomial
    u = tf.ones_like(prediction)

    for order in np.arange(1, library_config['poly_order']+1):
        u = tf.concat((u, u[:, order-1:order]*prediction), axis=1)
    u = tf.expand_dims(u, axis=2)

    # Gradients
    dy = tf.gradients(prediction, data)[0]
    y_t = dy[:, 1:2]
    y_x = dy[:, 0:1]

    du = tf.concat((tf.ones_like(y_x), y_x), axis=1)
    for order in np.arange(2, library_config['deriv_order']+1):
        du = tf.concat((du, tf.gradients(du[:, order-1], data)[0][:, 0:1]), axis=1)
    du = tf.expand_dims(du, axis=1)

    # Bringing it together
    theta = tf.matmul(u, du)
    theta = tf.reshape(theta, [tf.shape(theta)[0], tf.size(theta[0, :, :])])

    return [y_t], theta

def library_2Din_1Dout(data, prediction, library_config):
        '''
        Constructs a library graph in 1D. Library config is dictionary with required terms.
        '''

        # Polynomial
        u = tf.ones_like(prediction)

        for order in np.arange(1, library_config['poly_order']+1):
            u = tf.concat((u, u[:, order-1:order]*prediction), axis=1)
        u = tf.expand_dims(u, axis=2)

        # Gradients
        du = tf.gradients(prediction, data)[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = tf.gradients(u_x,data)[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = tf.gradients(u_y,data)[0][:, 2:3]
        du = tf.concat((tf.ones_like(u_x), u_x, u_y , u_xx, u_yy, u_xy), axis=1)
        #for order in np.arange(2, library_config['deriv_order']+1):
        #    du = tf.concat((du, tf.gradients(du[:, order-1], data)[0][:, 0:1]), axis=1)
        du = tf.expand_dims(du, axis=1)

        # Bringing it together
        theta = tf.matmul(u, du)
        theta = tf.reshape(theta, [tf.shape(theta)[0], tf.size(theta[0, :, :])])

        return [u_t], theta


def library_2Din_2Dout(data, prediction, library_config):

    #Polynomial in u
    u = tf.ones_like(prediction[:, 0:1])

    for order in np.arange(1, library_config['poly_order']+1):
        u = tf.concat((u, u[:, order-1:order]*prediction[:, 0:1]), axis=1)
    u = tf.expand_dims(u, axis=2)

    print(u.shape)

    # Polynomial in v
    v = tf.ones_like(prediction[:, 1:2])

    for order in np.arange(1, library_config['poly_order']+1):
        v = tf.concat((v, v[:, order-1:order]*prediction[:, 1:2]), axis=1)
    v = tf.expand_dims(v, axis=1)

    print(v.shape)
    # Calculating all cross terms
    uv = tf.matmul(u, v)
    uv = tf.reshape(uv, [tf.shape(u)[0], tf.size(uv[0, :, :])])
    uv = tf.expand_dims(uv,axis=2)


    print(uv.shape)
    # Derivative in u
    du = tf.gradients(prediction[:, 0:1], data)[0]
    u_t = du[:, 0:1]
    u_x = du[:, 1:2]
    u_y = du[:, 2:3]
    du2 = tf.gradients(u_x, data)[0]
    u_xx = du2[:, 1:2]
    u_xy = du2[:, 2:3]
    u_yy = tf.gradients(u_y, data)[0][:, 2:3]
    du = tf.concat((u_x, u_y , u_xx, u_yy, u_xy), axis=1)
    print(du.shape)
    # Derivative in v
    dv = tf.gradients(prediction[:, 1:2], data)[0]
    v_t = dv[:, 0:1]
    v_x = dv[:, 1:2]
    v_y = dv[:, 2:3]
    dv2 = tf.gradients(v_x, data)[0]
    v_xx = dv2[:, 1:2]
    v_xy = dv2[:, 2:3]
    v_yy = tf.gradients(v_y,data)[0][:, 2:3]
    dv = tf.concat((v_x, v_y , v_xx, v_yy, v_xy), axis=1)
    print(dv.shape)
    # Bringing du and dv together and calculating cross terms
    dudv = tf.concat((tf.ones_like(v_x), du, dv), axis=1)
    dudv = tf.expand_dims(dudv, axis=1)
    theta = tf.matmul(uv, dudv)
    theta = tf.reshape(theta, [tf.shape(theta)[0], tf.size(theta[0, :, :])])

    time_deriv = [u_t, v_t]

    return time_deriv, theta

def library_2Din_2Dout_lim(data, prediction, library_config):

    #Polynomial in u
    u = tf.ones_like(prediction[:, 0:1])

    for order in np.arange(1, library_config['poly_order']+1):
        u = tf.concat((u, u[:, order-1:order]*prediction[:, 0:1]), axis=1)
    u = tf.expand_dims(u, axis=2)


    # Polynomial in v
    v = tf.ones_like(prediction[:, 1:2])

    for order in np.arange(1, library_config['poly_order']+1):
        v = tf.concat((v, v[:, order-1:order]*prediction[:, 1:2]), axis=1)
    v = tf.expand_dims(v, axis=1)


    # Calculating all cross terms
    uv = tf.matmul(u, v)
    uv = tf.reshape(uv, [tf.shape(u)[0], tf.size(uv[0, :, :])])
    uv = tf.expand_dims(uv,axis=2)

    # Derivative in u
    du = tf.gradients(prediction[:, 0:1], data)[0]
    u_t = du[:, 0:1]
    u_x = du[:, 1:2]
    u_y = du[:, 2:3]
    du2 = tf.gradients(u_x, data)[0]
    u_xx = du2[:, 1:2]
    u_yy = tf.gradients(u_y, data)[0][:, 2:3]
    du = tf.concat((u_xx, u_yy), axis=1)

    # Derivative in v
    dv = tf.gradients(prediction[:, 1:2], data)[0]
    v_t = dv[:, 0:1]
    v_x = dv[:, 1:2]
    v_y = dv[:, 2:3]
    dv2 = tf.gradients(v_x, data)[0]
    v_xx = dv2[:, 1:2]
    v_yy = tf.gradients(v_y,data)[0][:, 2:3]
    dv = tf.concat((v_xx, v_yy), axis=1)

    # Bringing du and dv together and calculating cross terms
    dudv = tf.concat((tf.ones_like(v_x), du, dv), axis=1)
    dudv = tf.expand_dims(dudv, axis=1)
    theta = tf.matmul(uv, dudv)
    theta = tf.reshape(theta, [tf.shape(theta)[0], tf.size(theta[0, :, :])])

    time_deriv = [u_t, v_t]

    return time_deriv, theta

def library_1Din_2Dout_chemo(data, prediction, library_config):
    #Polynomial in u
    
    u = tf.ones_like(prediction[:, 0:1])
    for order in np.arange(1, library_config['poly_order']+1):
        u = tf.concat((u, u[:, order-1:order]*prediction[:, 0:1]), axis=1)
    u = tf.expand_dims(u, axis=2)

    print("u",u.shape)
    
    # Polynomial in v
    
    v = tf.ones_like(prediction[:, 1:2])
    for order in np.arange(1, library_config['poly_order']+1):
        v = tf.concat((v, v[:, order-1:order]*prediction[:, 1:2]), axis=1)
    v = tf.expand_dims(v, axis=1)

    print("v",v.shape)      
    
    # Derivative in u
    du = tf.gradients(prediction[:, 0:1], data)[0]
    u_t = du[:, 1:2]
    u_x = du[:, 0:1]
    
    du2 = tf.gradients(u_x, data)[0]
    u_xx = du2[:, 0:1]

    du = tf.concat((u_x,u_xx),axis=1)
    print("du",du.shape)
    
    # Derivative in v
    dv = tf.gradients(prediction[:, 1:2], data)[0]
    v_t = dv[:, 1:2]
    v_x = dv[:, 0:1]

    dv2 = tf.gradients(v_x, data)[0]
    v_xx = dv2[:, 0:1]
    
    dv=tf.concat((v_x, v_xx),axis=1)
    print("dv",dv.shape)
     
    #Calculating all cross derivative terms
    Du = du
    Dv = dv
    
    Du = tf.expand_dims(Du, axis=2)
    Dv = tf.expand_dims(Dv, axis=1)
    
    Ddudv = tf.matmul(Du,Dv)
    Ddudv = tf.reshape(Ddudv, [tf.shape(Ddudv)[0], tf.size(Ddudv[0, :, :])])
    
    # Calculating all cross terms
    
    uv = tf.matmul(u, v)
    uv = tf.reshape(uv, [tf.shape(u)[0], tf.size(uv[0, :, :])])
    uv = tf.expand_dims(uv,axis=2)

    print("uv",uv.shape)
    print("Ddudv",Ddudv.shape)
    
    # Bringing du and dv together and calculating cross terms
    dudv = tf.concat((tf.ones_like(v_x), du, dv, Ddudv), axis=1)
    
    print("dudv",dudv.shape)
    
    dudv = tf.expand_dims(dudv, axis=1)
    theta = tf.matmul(uv, dudv)
    theta = tf.reshape(theta, [tf.shape(theta)[0], tf.size(theta[0, :, :])])

    time_deriv = [u_t, v_t]

    return time_deriv, theta
  

