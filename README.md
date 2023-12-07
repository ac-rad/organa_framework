# Generating a Static Site from a Single Markdown File Using Pelican
Welcome to the guide on how to create your own static website using Pelican, a powerful static site generator. In this tutorial, we'll walk you through the process of turning a single Markdown file into a fully functional static site for your scientific paper.

## Prerequisites
Before we begin, ensure that you have the following installed on your computer:

- Python (3.7 or higher)
- Pip (Python package installer)

## Installation
1. Download the repository using the following command:

```bash
python -m pip install -r requirements.txt
```
## Setting Up Your Project

1. Create a new Markdown file inside the content directory.

2. Open the Markdown file you created and add the content of your scientific paper using the specified format. You can use the example provided as a template.

## Configuration

1. Open pelicanconf.py with a text editor and configure Pelican settings. Below is a minimal configuration example:

```python
AUTHOR = 'Your Name'
SITENAME = 'Your Site Name'
```
## Generating the Static Site
1. Open the terminal and navigate to your project directory.

2. Run the following command to generate the static site:

```bash
pelican content
```
Pelican will generate the static site in the output directory.

## Viewing Your Site Locally

### Invoke
The advantage of Invoke is that it is written in Python and thus can be used in a wide range of environments. The downside is that it must be installed separately. Use the following command to install Invoke, prefixing with sudo if your environment requires it:

```python
python -m pip install invoke
```

Take a moment to open the tasks.py file that was generated in your project root. You will see a number of commands, any one of which can be renamed, removed, and/or customized to your liking. Using the out-of-the-box configuration, you can generate your site via:
```python
invoke build
```
If you’d prefer to have Pelican automatically regenerate your site every time a change is detected (which is handy when testing locally), use the following command instead:
```python
invoke regenerate
```
To serve the generated site so it can be previewed in your browser at http://localhost:8000/:

```python
invoke serve
```
To serve the generated site with automatic browser reloading every time a change is detected, first `python -m pip install livereload`, then use the following command:

```python
invoke livereload
```

These are just a few of the commands available by default, so feel free to explore tasks.py and see what other commands are available. More importantly, don’t hesitate to customize tasks.py to suit your specific needs and preferences.


### Customizing the Site
You can further customize your static site by exploring Pelican's documentation and themes. Themes control the appearance and layout of your site. You can find themes at https://themes.getpelican.com/.

### Deploying Your Site
Once your static site is ready, you can deploy it to a web hosting service of your choice. You can use GitHub Pages, Netlify, or any other platform that supports static site hosting.

### Conclusion
Congratulations! You've successfully generated a static site for your scientific paper using Pelican. By following this guide, you've learned how to set up your project, configure Pelican, generate the static site, and even view it locally. Feel free to explore more advanced Pelican features to enhance your site's functionality and design. Happy static site building!