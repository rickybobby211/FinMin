#!/bin/bash

# Kontrollera att Docker är igång
if ! docker info > /dev/null 2>&1; then
  echo "Fel: Docker verkar inte vara igång."
  exit 1
fi

# Fråga efter Docker Hub användarnamn
read -p "Ange ditt Docker Hub användarnamn: " DOCKER_USER

if [ -z "$DOCKER_USER" ]; then
  echo "Du måste ange ett användarnamn."
  exit 1
fi

IMAGE_NAME="fingpt-forecaster-runpod"
FULL_IMAGE_NAME="$DOCKER_USER/$IMAGE_NAME:latest"

echo "Bygger Docker-image: $FULL_IMAGE_NAME..."
echo "Vi använder en bas-image med Flash Attention förinstallerat, så detta går snabbt!"

# Bygg imagen
docker build -t $FULL_IMAGE_NAME -f Dockerfile.runpod .

if [ $? -eq 0 ]; then
  echo "Bygget lyckades!"
  
  echo "Laddar upp till Docker Hub..."
  docker push $FULL_IMAGE_NAME
  
  if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------------"
    echo "Klar! Din template är nu uppladdad."
    echo ""
    echo "För att använda denna på RunPod:"
    echo "1. Gå till RunPod -> Templates -> New Template"
    echo "2. Image Name: $FULL_IMAGE_NAME"
    echo "3. Container Disk: 20GB (rekommenderas)"
    echo "4. Volume Disk: 50GB+ (för modeller och data)"
    echo "5. Mount Path: /workspace"
    echo "----------------------------------------------------------------"
  else
    echo "Uppladdning misslyckades. Är du inloggad? Kör 'docker login' först."
  fi
else
  echo "Bygget misslyckades."
fi
