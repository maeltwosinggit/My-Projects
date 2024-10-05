Hi there, this is a documentation for Project 7

===========
Error 1: ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed
How To Handle 1.1:  Usually happens after pip install requirements.txt on a freshly created a new Virtual Environment. Try to restart VS Code and run venv again.
How To Handle 1.2:  If there's a process "Reactivating Terminals", this might be the issue. Refer this link for fix https://stackoverflow.com/questions/78886125/vscode-python-extension-loading-forever-saying-reactivating-terminals

Error 2: ipynb cannot detect venv
How To Handle 2:    Usually because Python Global environment is installed in a different folder (try where python after source venv and see the path for global env and venv)
                    Go to VS Code Settings > Extensions > Python: Venv Path > set path to current project
                    Then Restart VS Code, it should detect the created venv https://stackoverflow.com/questions/58119823/jupyter-notebooks-in-visual-studio-code-does-not-use-the-active-virtual-environm