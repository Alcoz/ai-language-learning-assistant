# front

Interface Streamlit de type chatbot pour discuter avec le `conversation-engine`.

## Lancer

Démarrer d'abord le backend (depuis la racine du repo) :

```bash
uv run python conversation-engine/run_server.py
```

Puis le front :

```bash
uv run front
```

L'URL de l'API vaut `http://localhost:8000` par défaut. Elle se change via la variable
d'environnement `CONVERSATION_ENGINE_URL` ou directement dans la barre latérale.

## Voix

La barre de saisie accepte aussi un enregistrement micro : l'audio est envoyé à `/stt`
(faster-whisper) et c'est le texte transcrit qui apparaît dans la conversation.
Chaque réponse du prof est lue via `/tts` (Kokoro) et un lecteur audio reste sous son message.

Au premier lancement, le backend télécharge les modèles Whisper `small` et Kokoro-82M
depuis Hugging Face.
