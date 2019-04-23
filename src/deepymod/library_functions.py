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

def mech_library(data, prediction, library_config):
    '''
    Constructs a library graph in 1D. Library config is dictionary with required terms.
    '''
    y_t = tf.gradients(prediction, data)[0]    
    y_tt   = tf.gradients(y_t, data)[0]
    y_ttt   = tf.gradients(y_tt, data)[0]
    y_tttt   = tf.gradients(y_ttt, data)[0]
    
    sigma0 =1;T=1/4
    sigma = sigma0*(1-tf.exp(-data/T))         # increasing stress variable, speed controlled by T
    sigma = tf.sin(data)/data
    sigma_t = (data*tf.cos(data) - tf.sin(data))/(data**2)
    sigma_tt = -((data**2-2)*tf.sin(data)+2*data*tf.cos(data))/(data**3)
    sigma_ttt = (3*(data**2-2)*tf.sin(data)-data*(data**2-6)*tf.cos(data))/(data**4)
    sigma_tttt = (4*(data**2-6)*tf.cos(data)+(data**4-12*data**2+24)*tf.sin(data))/(data**5)
    sigma_ttttt = (data*(data**4-20*data**2+120)*tf.cos(data)-5*(data**4-12*data**2+24)*tf.sin(data))/(data**6)
    theta = tf.concat((sigma, sigma_t, sigma_tt,sigma_ttt, prediction, y_tt,y_ttt), axis=1)
    return [y_t], theta

def generalized_maxwell(data, prediction, library_config):
    # Creating libraries
    sigma_library = library_config['stress_function'](data, library_config['stress_function_args']) # time is first column
    epsilon_library = prediction    
    
    for order in np.arange(1, library_config['max_order']+1):
        sigma_t =  tf.gradients(sigma_library[:, order-1:order], data)[0]
        epsilon_t =  tf.gradients(epsilon_library[:, order-1:order], data)[0]
        
        sigma_library = tf.concat([sigma_library, sigma_t], axis=1)
        epsilon_library = tf.concat([epsilon_library, epsilon_t], axis=1)
        
    # Taking out the first derivative of epsilon
    mask = tf.one_hot(tf.ones(tf.shape(epsilon_library)[0], dtype=tf.int32), tf.shape(epsilon_library)[1], on_value=1, off_value=0, axis=-1) 
    epsilon_library, epsilon_t = tf.dynamic_partition(epsilon_library, mask, 2)
    epsilon_library = tf.reshape(epsilon_library, [tf.shape(prediction)[0], library_config['max_order']]) # size before was max_order+1
    
    library = tf.concat([sigma_library,  epsilon_library], axis=1)
    time_deriv = [tf.expand_dims(epsilon_t, axis=1)]
    
    return time_deriv, library
    
    

    

  

