# Backend Project

## Introduction
This is the backend project for the Visual AI Assistant. It provides the necessary APIs and services to support the frontend application.

## Prerequisites
- Python (version 3.8 or higher)
- pip (version 20 or higher)
- MongoDB (version 4 or higher)

## Installation
1. Clone the repository
2. Navigate to the backend directory
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    ```bash
    # On Windows
    venv\Scripts\activate

    # On macOS/Linux
    source venv/bin/activate
    ```
5. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration
1. Create a `.env` file in the root of the backend directory and add the following environment variables:
    ```env
    OPENAI_API_KEY
    ```

## Running the Application
1. Start the backend server:
    ```bash
    uvicorn main:app --reload
    ```
2. The server should now be running at `http://localhost:8000`.

## Contributing
Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or feedback, please contact `duvvana.vamsi@gmail.com`.

