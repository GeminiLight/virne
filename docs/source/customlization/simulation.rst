Simulation Settings
===================

General Settings
----------------

Physical Network
~~~~~~~~~~~~~~~~

To simulate various physical networks, a highly customizable setting file `p_net_setting.yaml` for physical network is provided, where you can update parameters to model different setups.
This configuration file is used to specify the settings for the physical network. The following parameters can be configured:

- `num_nodes`: the number of nodes in the physical network.
- `save_dir`: the directory in which to save the output file.
- `topology`: the topology to be used for the physical network. This can either be a file path to a .gml file or one of several built-in network models. In this case, a Waxman model is used with parameters wm_alpha: 0.5 and wm_beta: 0.2.
- `link_attrs_setting`: the attributes to be assigned to links in the physical network. In this case, a single attribute is specified: bw, which represents the link bandwidth. It is assigned a uniform distribution with minimum value 50 and maximum value 100.
- `node_attrs_setting`: the attributes to be assigned to nodes in the physical network. In this case, a single attribute is specified: cpu, which represents the node's processing power. It is assigned a uniform distribution with minimum value 50 and maximum value 100.
- `file_name`: the name of the output file to be generated, which will be saved in the save_dir directory. In this case, the file name is p_net.gml.

.. note:: 
    
    The `link_attrs_setting` and `node_attrs_setting` parameters can be commented out as needed, and additional attributes can be added if desired.


Virtual Network Requests
~~~~~~~~~~~~~~~~~~~~~~~~

To simulate various virtual network simulator, a highly customizable setting file `v_smi_setting.yaml` for virtual network simulator is provided, where you can update parameters to model different setups.
This configuration file is used to specify the settings for the virtual network simulator. The following parameters can be configured:

- `num_nodes`: The number of nodes in the virtual network.
- `save_dir`: The directory where the resulting virtual network topology will be saved.
- `topology`: The topology of the virtual network. This can either be a path to a file in GML format, or a randomly generated topology using the Waxman model. If the Waxman model is used, the wm_alpha and wm_beta parameters specify the model parameters.
- `link_attrs_setting`: The attributes of the links in the virtual network. The distribution, data type, and value range of each attribute can be specified, along with whether the values should be randomly generated or loaded from an external file. The name, owner, and type of each attribute must also be specified.
- `node_attrs_setting`: The attributes of the nodes in the virtual network. The distribution, data type, and value range of each attribute can be specified, along with whether the values should be randomly generated or loaded from an external file. The name, owner, and type of each attribute must also be specified.
- `file_name`: The name of the file where the resulting virtual network topology will be saved, in GML format.

.. note:: 
    
    The `link_attrs_setting` and `node_attrs_setting` parameters can be commented out as needed, and additional attributes can be added if desired.


Topology Settings
-----------------
