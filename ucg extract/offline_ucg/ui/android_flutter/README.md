# Android client (Flutter)

Thin client for the FastAPI service. The model and database stay on the clinic computer; the phone/tablet connects over
the local Wi-Fi or a phone hotspot, so nothing needs the internet.

    flutter create --platforms=android .      # generates android/ once (keeps lib/ and pubspec.yaml)
    flutter pub get && flutter run

Plain-HTTP on a LAN needs cleartext traffic allowed: in android/app/src/main/AndroidManifest.xml add to <application>
    android:usesCleartextTraffic="true"
Tap the network icon in the app bar to set the server address (e.g. http://192.168.1.20:8000).
Start the server with:  uvicorn app:app --host 0.0.0.0 --port 8000
