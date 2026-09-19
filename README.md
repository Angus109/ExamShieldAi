_Professor Dashboard_

- AI-powered question generation for objective & subjective questions
- Exam creation, history, and sharing with students
- Question management (view, update, delete)
- Marks insertion and result publication
- Live and automatic exam monitoring using WebRTC/TensorFlow for secure proctoring
- Problem reporting and 24/7 support
- Exam wallet recharge

_Student Dashboard_

- Take exams, view exam history, and check results
- Report problems
- Features:
  - Negative marking
  - Randomized questions
  - Calculator for math-based exams
  - 20+ compilers/interpreters for programming practical exams

_Security Features_

- Facial recognition technology for impersonation prevention
- Advanced proctoring with:
  - Window event logging (tab changes, new tabs)
  - Audio frequency logging (every 5 seconds)
  - Mobile phone detection
  - Multi-person detection
  - Gaze estimation (body & eye movement tracking)
  - Image logging (every 5 seconds)
  - Disabled functions (cut, copy, paste, screenshots)
  - VM detection and screen-sharing application detection

RUNNING WITH DOCKER
sudo docker stop flask-container
sudo docker rm flask-container
sudo docker build -t my-flask-app .
sudo docker run -d -p 5000:5000 --name flask-container my-flask-app
