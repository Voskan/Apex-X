FROM golang:1.24-alpine AS build

WORKDIR /src

COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal

RUN go test ./...
RUN CGO_ENABLED=0 go build -o /out/apexx-runtime ./cmd/apexx-runtime

FROM alpine:3.21

RUN addgroup -S app && adduser -S app -G app
USER app

COPY --from=build /out/apexx-runtime /usr/local/bin/apexx-runtime

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/apexx-runtime"]
CMD ["-addr",":8080","-adapter","onnxruntime","-default-budget-profile","balanced"]

