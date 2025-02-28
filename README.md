
## Project Structure
AyakaUtils/
├── LICENSE
├── README.md
├── requirements.txt        # for development/testing dependencies
├── setup.py                # packaging script
├── ayaka_utils/            # main package
│   ├── __init__.py         # makes it a package; you can import utilities directly from here
│   ├── Classes/            # Classes for Pydantic models
│   ├── Defs/               # Python functions
│   ├── Other/              # Other utility modules
│   └── Runnables/          # Runnable functions for LangChain
└── tests/                  # Unit tests
    └── TEST-BuildReadableEmoDisc.py
