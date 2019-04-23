from deepymod.utilities import tensorboard_to_dataframe

event_file = 'Examples/output/burgers/2019-04-23_17:17:18/iteration_0/events.out.tfevents.1556032642.C02QP0AZG8WL.local'
df_tb = tensorboard_to_dataframe(event_file)

print(df_tb.keys())
