## eastbay


    Add a short description here!



## Troubleshooting / FAQ

*Q*. `eb.fit` displays a large blank space instead of a plot when running in Jupyter notebook.
*A*. This can occur if you are running Jupyter on a different host and using
     SSH tunneling. The plotting code uses [Plotly dash](http://dash.plotly.com), which
     starts a separate webserver on part 8050 to handle plotting. To fix it,
     add a local forward on this port (`-L 8050:localhost:8050`) to your ssh
     login command.


### Making Changes & Contributing

Contributions to improve the project are always welcome.

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd eastbay
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate
