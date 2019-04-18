from tensorboard.summary import custom_scalar_pb
from tensorboard.plugins.custom_scalar import layout_pb2
import tensorflow as tf
import numpy as np

def tb_setup(graph, output_opts):
    # Cost summaries
    [tf.summary.scalar('MSE_cost_'+str(idx), cost) for idx, cost in enumerate(tf.unstack(graph.MSE_costs))]
    [tf.summary.scalar('PI_cost_'+str(idx), cost) for idx, cost in enumerate(graph.PI_costs)]
    [tf.summary.scalar('L1_cost_'+str(idx), cost) for idx, cost in enumerate(graph.L1_costs)]
    tf.summary.scalar('Total_cost', graph.loss)
    tf.summary.scalar('Loss_Grad', graph.gradloss)

    # Coefficient summaries
#    [[tf.summary.scalar('Coeff_' + str(idx_coeff) + '_Comp_' + str(idx_comp), tf.squeeze(comp)) for idx_comp, comp in enumerate(tf.unstack(coeff, axis=0))] for idx_coeff, coeff in enumerate(tf.unstack(graph.coeff_list))]
    [[tf.summary.scalar('Coeff_' + str(idx_coeff) + '_Comp_' + str(idx_comp), tf.squeeze(comp)) for idx_comp, comp in enumerate(tf.unstack(coeff, axis=0))] for idx_coeff, coeff in enumerate(graph.coeff_list)]
    [[tf.summary.scalar('Scaled_Coeff_' + str(idx_coeff) + '_Comp_' + str(idx_comp), tf.squeeze(comp)) for idx_comp, comp in enumerate(tf.unstack(coeff, axis=0))] for idx_coeff, coeff in enumerate(graph.coeff_scaled_list)]

    # Merging everything and making custom board
    merged_summary = tf.summary.merge_all()
    custom_board = custom_board_generator(graph)

    return merged_summary, custom_board

def custom_board_generator(graph):
    # We make the coefficient and scaled coefficient charts first because we need to do it dynamically.
    coeff_chart = [layout_pb2.Chart(title='Coeff_' + str(idx), multiline=layout_pb2.MultilineChartContent(tag=[r'Coeff_' + str(idx) + '_Comp_*'])) for idx in np.arange(len(graph.PI_costs))]
    coeff_scaled_chart = [layout_pb2.Chart(title='Scaled_Coeff_' + str(idx), multiline=layout_pb2.MultilineChartContent(tag=[r'Scaled_Coeff_' + str(idx) + '_Comp_*'])) for idx in np.arange(len(graph.PI_costs))]

    # Actually making the board
    custom_board = custom_scalar_pb(
        layout_pb2.Layout(category=[
            layout_pb2.Category(title='Training',
                chart=[layout_pb2.Chart(title='MSE_Losses', multiline=layout_pb2.MultilineChartContent(tag=[r'MSE_cost_*'])),
                       layout_pb2.Chart(title='PI_Losses', multiline=layout_pb2.MultilineChartContent(tag=[r'PI_cost_*'])),
                       layout_pb2.Chart(title='L1_Losses', multiline=layout_pb2.MultilineChartContent(tag=[r'L1_cost_*'])),
                       layout_pb2.Chart(title='Total_cost', multiline=layout_pb2.MultilineChartContent(tag=['Total_cost'])),
                       layout_pb2.Chart(title='Gradloss', multiline=layout_pb2.MultilineChartContent(tag=['Loss_Grad']))\
                        ]
            ),
        layout_pb2.Category(title='Coefficients', chart=coeff_chart),
        layout_pb2.Category(title='Scaled_Coefficients', chart=coeff_scaled_chart)
            ]))

    return custom_board
