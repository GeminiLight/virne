SFC Deployment
==============

Virne also is easily adaptable to a SFC deployment by modifying the topology setting of the VNR simulator. 

In SFC Deployment problem, the topology is defined as a chain-style graph, i.e., the path graph topology. 

The topology setting of VNR simulator is defined in the file **v_sim_setting.yaml** in the **settings** folder. 

Therefore, the topology setting of the VNR simulator can be modified as follows:

.. code-block:: yaml

    topology: 
        type: path
    