
# AI for medical imaging course
This GIT contains the exercises and examples from the UvA/VU AI for medical imaging course.
Please perform the exercise with your own code. You are not allowed to copy (large chuncks) of code from fellow students, from the internet, or from elsewhere.
Please do not share your code or make your code public.
You are allowed to help fellow students by nudging them in the right direction.

## Getting started
Here are some beginner friendly steps to get started with the assignments. 

### 1.	Download a code editor
_Why? A code editor is useful for writing, editing, and managing code efficiently. Just like you need the program Word to open a .docx document, you need an editor to open a .py or .ipynb document._

In this walkthrough I will be working with PyCharm, which can be downloaded [here](https://www.jetbrains.com/pycharm/download/?section=mac). You can set up a [student account](https://www.jetbrains.com/community/education/#students) so that you have a free license to use it, or just use the free trial. 

Feel free to download or use the editor you like, examples are [Spyder](https://www.spyder-ide.org/) and [VS code](https://code.visualstudio.com/download).

### 2.	Download the codebase of the course
_Why? You need to do this step so that you don’t have to cut and paste the code, we designed easy to use notebooks for you to make running the code easy._

You are currently in the right GitHub project. To download the code, click on the green button <> code. Then, click Download ZIP. Unpack the zipfile in a folder where you want to put your code. 

Note that you can also [git clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the repository, if that is more familiar to you. 

### 3.	Open the codebase with your editor
Now open your editor, in this walkthrough I am using PyCharm. Click file > open > Then choose the project map you have just downloaded. 

This will open the full folder in your code editor, instead of just one file. This is needed so that you have everything in one place, and files that depend on other files can access each other. 

### 4.	Create a virtual environment
_Why? Using virtual environments in Python is essential for managing package dependencies and avoiding conflicts between different projects. Ideally, you create a new python environment for every project you do._

When I open the project in PyCharm, I get a pop up asking if I want to create an environment. Here, I click OK. A new folder has now been created called .venv. 
In the bottom right corner, you will see \<No interpreter>. Click on that, and select the python environment you have just created.

If you don’t get this pop up, that’s fine, we will walk through it manually.  You still want to click on the  \<no interpreter> button in the bottom right corner. You need to click on Add new interpreter > add local interpreter. Click OK. 

Note that you can also create a virtual environment through a terminal. You can do this with [venv](https://docs.python.org/3/library/venv.html) or [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### 5.	Install the requirements
_What and why? Requirements.txt is a file in the codebase that contains the names of packages that are needed to run the code that we have provided for you. Since you just created a new environment, it means that you don’t have any packages at all and we need to install them._

In order to install the requirements, please open a terminal. Every code editor has a different way of opening a terminal so you might need to google it. In PyCharm you click view > tool windows > terminal. 

In the bottom of your editor, a terminal is now opened. Here, you need to type: 

```pip install -r requirements.txt```

and run it.

If you get an error, the solution is to go to your requirements.txt file and remove the version numbers. Save the file, and run the command pip install -r requirements.txt in your terminal again. It should now start to download the packages. 

### 6.	Open the assignment
Once everything is installed, it means you can try to run the code! In your folder structure, open assignment 1 > exercise_1.ipynb. You see that a notebook has now been opened. Here, the exercise is described again and it also contains the code. 

If you scroll down, try to run the first codeblock that contains all the imports. If all goes well, you see a green check. It means that you are now good to go! Good luck with the exercises :) 





