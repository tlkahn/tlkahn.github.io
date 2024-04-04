+++
title = "How I build a meditation app in a week"
description = "Dabbling into iOS dev"
date = 2023-12-25T21:15:12+08:00
tags = ["programming", "meditation"]
menu = "main"
+++

## Prelude

The story was I built an iOS app for group meditation. It works like [Clubhouse](https://clubhouse.com) but for regular meditators instead. The instructor can remotely sit with students, who can reserve class and book 1:1 sessions. Teachers can record sessions and offer the replay option for students. During meditation, they can play ambient music along with guidance.

I built the app in a week. I have to say that Swift is a much more enjoyable language than Objective-C which I developed apps on years ago. Here is a high level overview of the short story. You can find the complete code on [Github](https://github.com/tlkahn/vima-ios).

My apologies that I had little time to write more documentation for it. Fortunately Swift is expressive enough and I had made considerable efforts making my code as readable as possible.

## Streaming

I use Agora as the audio streaming vendor (like Clubhouse). The streaming service was wrapped into a class. The reason is mostly for compatibility with Agora's older SDKs. Also this simplified referencing and potential inheritance. The class structure appears as follows:

```swift
class Broadcaster: NSObject, AgoraRtcEngineDelegate, ObservableObject {
    var channelName: String
    var uid: UInt
    var role: AgoraClientRole
    var recordingConfig: AgoraAudioRecordingConfiguration?
    @State private var cancellables = Set<AnyCancellable>()
    @State private var initialized: Bool = false
    @State private var joined: Bool = false
    private(set) var token: String?
    private var bags = Set<AnyCancellable>()
    var agoraKit: AgoraRtcEngineKit!

    var connectionState: AgoraConnectionState {
        agoraKit.getConnectionState()
    }

    init(channelName: String, role: AgoraClientRole, uid: UInt) {
        self.channelName = channelName
        self.role = role
        self.uid = uid
        super.init()
    }
```

## Network Service

Swift code for networking is almost boilerplate thanks to Swift's versatile and modern language design.

```swift
struct NetworkService {
    let baseURL: String
    private func getToken() -> String? {
        return UserDefaults.standard.object(forKey: "token") as? String
    }

    func get<U>(from: String) -> AnyPublisher<U, Error> where U: Decodable {
        let url = URL(string: baseURL + from)!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        if getToken() != nil {
            request.setValue("Bearer \(getToken()!)", forHTTPHeaderField: "Authorization")
        }
        return run(request)
    }

    func post<T, U>(_ entry: T, to: String) -> AnyPublisher<U, Error>
        where T: Encodable, U: Decodable
    {
        let url = URL(string: baseURL + to)!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        if getToken() != nil {
            request.setValue("Bearer \(getToken()!)", forHTTPHeaderField: "Authorization")
        }
        let encoder = JSONEncoder()
        let jsonData = try! encoder.encode(entry)
        request.httpBody = jsonData
        return run(request)
    }

    func run<T: Decodable>(_ request: URLRequest) -> AnyPublisher<T, Error> {
        let decoder = JSONDecoder()
        var result: AnyPublisher<T,Error>
        do {
            result = URLSession.shared
                .dataTaskPublisher(for: request)
                .map { $0.data }
                .handleEvents(receiveOutput: { print("<<< Data received:\n", NSString(
                    data: $0,
                    encoding: String.Encoding.utf8.rawValue
                )!) })
                .decode(type: T.self, decoder: decoder)
                .receive(on: DispatchQueue.main)
                .eraseToAnyPublisher()
        }
        return result
    }
}
```

Syncing heartbeat with servers using websocket is also straightforward:

```swift
func sendHeartbeat() {
    let message = URLSessionWebSocketTask.Message.string("heartbeat")
    webSocketTask?.send(message) { error in
        if let error = error {
            print("Error sending heartbeat: \(error)")
        }
    }
}
```

## UI

SwiftUI is declarative, almost like CSS, which quite pleasantly surprised me. I was still traumatized by Obj-C UI experience. This is even more sweetened by `extension` syntax to separate UI from event handling and state transitions. You can write those code in a breeze:

```swift
extension MusicListView {

    func play(music: Payload.Music?) {
        guard let music = music, let url = URL(string: music.url) else { return }

        if music == musicInPlay, let player = audioPlayer, !isPlaying {
            player.play()
        } else {
            stop()
            let playerItem = AVPlayerItem(url: url)
            audioPlayer = AVPlayer(playerItem: playerItem)

            // Resume playback from the stored played time
            if let storedProgress = playbackProgress[music] {
                audioPlayer?.seek(to: CMTime(
                    seconds: storedProgress.played,
                    preferredTimescale: 1
                ))
            }
            audioPlayer?.play()
            musicInPlay = music
        }
        isPlaying = true
    }

    func stop() {
        audioPlayer?.pause()
        audioPlayer = nil
        musicInPlay = nil
        isPlaying = false
    }

    func pause() {
        audioPlayer?.pause()
        isPlaying = false
    }
```

## Reflections

- You might well want to work with Swift, but not Apple. Maybe Rust offers such circumvention with more powerful browser support and embedded wasm code.
- The paradigm on mobile dev today has very closely resembled front-end development. I was constantly reminded of react/redux, and such.
- Please let me know if you find this app useful.
