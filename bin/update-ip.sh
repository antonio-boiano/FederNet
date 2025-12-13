docker pull busybox:stable

CONTAINER=$(docker create busybox:stable)
docker cp "$CONTAINER":/bin/busybox ./bin/busybox
docker rm "$CONTAINER"
mv ./bin/busybox ./bin/ip
chmod +x ./bin/ip