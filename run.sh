#!/usr/bin/env sh
# Starting the Streamlit app
python -m streamlit run navigator.py --server.enableXsrfProtection false > /dev/null 2>&1 &