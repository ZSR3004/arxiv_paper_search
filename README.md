# Arxiv Paper Search

## Table of Contents

- [Project Brief](#project-brief)
- [Project Setup](#project-setup)
- [Usage](#usage)

## Project Brief

This project uses the [arXiv API](https://info.arxiv.org/help/api/index.html) to find academic papers that are similar based on a keyword search.

[arXiv](https://arxiv.org/) is a repository of academic papers from various disciplines including physics, computer science, mathematics, and more.

A full example is located in the `quantum_example.ipynb` notebook.

This project is not affiliated with or endorsed by arXiv. Thank you to arXiv for use of its open access interoperability.

## Project Setup

Start by cloning this repository:

```bash
git clone https://github.com/ZSR3004/arxiv_paper_search
cd arxiv_paper_search
```

You'll also need to create a `.env` file. Inside, write the following.

```bash
api_key = "123abc" # or whatever your gemini API key is
```

Now, assuming you already have Python 3 installed, create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Installed required dependencies.

```bash
pip install numpy pandas python-dotenv google-generativeai
```

## Usage

The main function to use is: `find_similar_papers`.

This will search Arxiv for papers based on your query and find similar ones using embeddings. You can ignore the rest of the repository if you're only interested in using the program.
