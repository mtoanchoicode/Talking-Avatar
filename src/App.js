import React, { Suspense, useEffect, useRef, useState, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  useGLTF,
  useTexture,
  Loader,
  Environment,
  useFBX,
  useAnimations,
  OrthographicCamera,
} from "@react-three/drei";
import { MeshStandardMaterial } from "three/src/materials/MeshStandardMaterial";

import { LineBasicMaterial, MeshPhysicalMaterial, Vector2 } from "three";
import ReactAudioPlayer from "react-audio-player";

import createAnimation from "./converter";
import blinkData from "./blendDataBlink.json";
import thinkingData from "./thinkingData.json";

import * as THREE from "three";
import axios from "axios";
import { SRGBColorSpace, LinearSRGBColorSpace } from "three";

const _ = require("lodash");

const host = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

// Thinking animation data - subtle head nod and thinking expression

function Avatar({
  avatar_url,
  speak,
  setSpeak,
  text,
  setAudioSource,
  playing,
  isThinking,
  blendData,
}) {
  let gltf = useGLTF(avatar_url);
  let morphTargetDictionaryBody = null;
  let morphTargetDictionaryLowerTeeth = null;

  const [
    bodyTexture,
    eyesTexture,
    teethTexture,
    bodySpecularTexture,
    bodyRoughnessTexture,
    bodyNormalTexture,
    teethNormalTexture,
    hairTexture,
    tshirtDiffuseTexture,
    tshirtNormalTexture,
    tshirtRoughnessTexture,
    hairAlphaTexture,
    hairNormalTexture,
    hairRoughnessTexture,
  ] = useTexture([
    "/images/body.webp",
    "/images/eyes.webp",
    "/images/teeth_diffuse.webp",
    "/images/body_specular.webp",
    "/images/body_roughness.webp",
    "/images/body_normal.webp",
    "/images/teeth_normal.webp",
    "/images/h_color.webp",
    "/images/tshirt_diffuse.webp",
    "/images/tshirt_normal.webp",
    "/images/tshirt_roughness.webp",
    "/images/h_alpha.webp",
    "/images/h_normal.webp",
    "/images/h_roughness.webp",
  ]);

  _.each(
    [
      bodyTexture,
      eyesTexture,
      teethTexture,
      teethNormalTexture,
      bodySpecularTexture,
      bodyRoughnessTexture,
      bodyNormalTexture,
      tshirtDiffuseTexture,
      tshirtNormalTexture,
      tshirtRoughnessTexture,
      hairAlphaTexture,
      hairNormalTexture,
      hairRoughnessTexture,
    ],
    (t) => {
      t.colorSpace = SRGBColorSpace;
      t.flipY = false;
    }
  );

  bodyNormalTexture.colorSpace = LinearSRGBColorSpace;
  tshirtNormalTexture.colorSpace = LinearSRGBColorSpace;
  teethNormalTexture.colorSpace = LinearSRGBColorSpace;
  hairNormalTexture.colorSpace = LinearSRGBColorSpace;

  gltf.scene.traverse((node) => {
    if (
      node.type === "Mesh" ||
      node.type === "LineSegments" ||
      node.type === "SkinnedMesh"
    ) {
      node.castShadow = true;
      node.receiveShadow = true;
      node.frustumCulled = false;

      if (node.name.includes("Body")) {
        node.castShadow = true;
        node.receiveShadow = true;

        node.material = new MeshPhysicalMaterial();
        node.material.map = bodyTexture;
        node.material.roughness = 1.7;

        node.material.roughnessMap = bodyRoughnessTexture;
        node.material.normalMap = bodyNormalTexture;
        node.material.normalScale = new Vector2(0.6, 0.6);

        morphTargetDictionaryBody = node.morphTargetDictionary;

        node.material.envMapIntensity = 0.8;
      }

      if (node.name.includes("Eyes")) {
        node.material = new MeshStandardMaterial();
        node.material.map = eyesTexture;
        node.material.roughness = 0.1;
        node.material.envMapIntensity = 0.5;
      }

      if (node.name.includes("Brows")) {
        node.material = new LineBasicMaterial({ color: 0x000000 });
        node.material.linewidth = 1;
        node.material.opacity = 0.5;
        node.material.transparent = true;
        node.visible = false;
      }

      if (node.name.includes("Teeth")) {
        node.receiveShadow = true;
        node.castShadow = true;
        node.material = new MeshStandardMaterial();
        node.material.roughness = 0.1;
        node.material.map = teethTexture;
        node.material.normalMap = teethNormalTexture;

        node.material.envMapIntensity = 0.7;
      }

      if (node.name.includes("Hair")) {
        node.material = new MeshStandardMaterial();
        node.material.map = hairTexture;
        node.material.alphaMap = hairAlphaTexture;
        node.material.normalMap = hairNormalTexture;
        node.material.roughnessMap = hairRoughnessTexture;

        node.material.transparent = true;
        node.material.depthWrite = false;
        node.material.side = 2;
        node.material.color.setHex(0x000000);

        node.material.envMapIntensity = 0.3;
      }

      if (node.name.includes("TSHIRT")) {
        node.material = new MeshStandardMaterial();

        node.material.map = tshirtDiffuseTexture;
        node.material.roughnessMap = tshirtRoughnessTexture;
        node.material.normalMap = tshirtNormalTexture;
        node.material.color.setHex(0xffffff);

        node.material.envMapIntensity = 0.5;
      }

      if (node.name.includes("TeethLower")) {
        morphTargetDictionaryLowerTeeth = node.morphTargetDictionary;
      }
    }
  });

  const [clips, setClips] = useState([]);
  const mixer = useMemo(() => new THREE.AnimationMixer(gltf.scene), []);
  const thinkingActionRef = useRef(null);

  // New effect: react to incoming blendData
  useEffect(() => {
    if (!blendData || !morphTargetDictionaryBody) return;

    const newClips = [
      createAnimation(blendData, morphTargetDictionaryBody, "HG_Body"),
      createAnimation(
        blendData,
        morphTargetDictionaryLowerTeeth,
        "HG_TeethLower"
      ),
    ];

    setClips(newClips);
    // setBlendData(null); // optional: clear after use to prevent replay
  }, [blendData, morphTargetDictionaryBody, morphTargetDictionaryLowerTeeth]);

  let idleFbx = useFBX("/idle.fbx");
  let { clips: idleClips } = useAnimations(idleFbx.animations);

  idleClips[0].tracks = _.filter(idleClips[0].tracks, (track) => {
    return (
      track.name.includes("Head") ||
      track.name.includes("Neck") ||
      track.name.includes("Spine2")
    );
  });

  idleClips[0].tracks = _.map(idleClips[0].tracks, (track) => {
    if (track.name.includes("Head")) {
      track.name = "head.quaternion";
    }

    if (track.name.includes("Neck")) {
      track.name = "neck.quaternion";
    }

    if (track.name.includes("Spine")) {
      track.name = "spine2.quaternion";
    }

    return track;
  });

  useEffect(() => {
    let idleClipAction = mixer.clipAction(idleClips[0]);
    idleClipAction.play();

    let blinkClip = createAnimation(
      blinkData,
      morphTargetDictionaryBody,
      "HG_Body"
    );
    let blinkAction = mixer.clipAction(blinkClip);
    blinkAction.play();
  }, []);

  // Handle thinking animation
  useEffect(() => {
    if (isThinking && morphTargetDictionaryBody) {
      // Stop any existing thinking animation
      if (thinkingActionRef.current) {
        thinkingActionRef.current.stop();
      }

      // Create and play thinking animation
      let thinkingClip = createAnimation(
        thinkingData,
        morphTargetDictionaryBody,
        "HG_Body"
      );

      if (thinkingClip) {
        thinkingActionRef.current = mixer.clipAction(thinkingClip);
        thinkingActionRef.current.setLoop(THREE.LoopRepeat);
        thinkingActionRef.current.play();
      }
    } else if (!isThinking && thinkingActionRef.current) {
      // Stop thinking animation when not thinking
      thinkingActionRef.current.fadeOut(0.5);
      setTimeout(() => {
        if (thinkingActionRef.current) {
          thinkingActionRef.current.stop();
        }
      }, 500);
    }
  }, [isThinking, morphTargetDictionaryBody]);

  // Play speech animation clips when available
  useEffect(() => {
    if (playing === false) return;

    // Stop thinking animation when starting to speak
    if (thinkingActionRef.current) {
      thinkingActionRef.current.stop();
    }

    _.each(clips, (clip) => {
      let clipAction = mixer.clipAction(clip);
      clipAction.setLoop(THREE.LoopOnce);
      clipAction.play();
    });
  }, [playing]);

  useFrame((state, delta) => {
    mixer.update(delta);
  });

  return (
    <group name="avatar">
      <primitive object={gltf.scene} dispose={null} />
    </group>
  );
}

function makeSpeech(text) {
  return axios.post(host + "/talk", { text });
}

const STYLES = {
  area: { position: "absolute", bottom: "10px", left: "10px", zIndex: 500 },
  text: {
    margin: "0px",
    width: "300px",
    padding: "5px",
    background: "none",
    color: "#ffffff",
    fontSize: "1.2em",
    border: "none",
  },
  speak: {
    padding: "10px",
    marginTop: "5px",
    display: "block",
    color: "#FFFFFF",
    background: "#222222",
    border: "None",
    cursor: "pointer",
  },
  voiceButton: {
    padding: "10px",
    marginTop: "5px",
    display: "block",
    color: "#FFFFFF",
    background: "#444444",
    border: "None",
    cursor: "pointer",
  },
  area2: { position: "absolute", top: "5px", right: "15px", zIndex: 500 },
  label: { color: "#777777", fontSize: "0.8em" },
  status: { color: "#00ff00", fontSize: "0.9em", marginTop: "5px" },
};

function App() {
  const audioPlayer = useRef();
  const recognitionRef = useRef(null);

  const [speak, setSpeak] = useState(false);
  const [text, setText] = useState("Hello how are you today");
  const [audioSource, setAudioSource] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [blendData, setBlendData] = useState(null);

  // Session management
  const [sessionId, setSessionId] = useState(() => {
    // Generate initial session on load
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  });

  // Mode management (RAG vs LLM-only)
  const [useRAG, setUseRAG] = useState(true);

  // Initialize speech recognition
  useEffect(() => {
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = "en-US";

      recognitionRef.current.onstart = () => {
        setIsListening(true);
        setVoiceStatus("Listening...");
      };

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setText(transcript.substring(0, 200));
        setVoiceStatus("Processing...");

        // Auto-send to backend after voice recognition
        setTimeout(() => {
          setSpeak(true);
          setIsThinking(true);
        }, 500);
      };

      recognitionRef.current.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
        setVoiceStatus(`Error: ${event.error}`);
        setTimeout(() => setVoiceStatus(""), 3000);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
        if (voiceStatus === "Listening...") {
          setVoiceStatus("");
        }
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  // Monitor speak state to control thinking animation
  useEffect(() => {
    if (speak && !playing) {
      setIsThinking(true);
      setVoiceStatus("Thinking...");
    }
  }, [speak, playing]);

  // Modified makeSpeech to include session and mode
  const makeSpeechWithSession = (text) => {
    return fetch(host + "/talk", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        session_id: sessionId,
        use_rag: useRAG,
      }),
    }).then((res) => res.json());
  };

  // Override speak effect to use session
  useEffect(() => {
    if (speak === false) return;

    setIsThinking(true); // optional: show thinking while waiting

    makeSpeechWithSession(text)
      .then((response) => {
        let { blendData, filename } = response; // make sure your backend still returns blendData!

        filename = host + filename;

        setBlendData(blendData); // Save blendData
        setAudioSource(filename); // Set audio
        setSpeak(false); // important: reset speak trigger
      })
      .catch((err) => {
        console.error(err);
        setSpeak(false);
        setIsThinking(false);
      });
  }, [speak, sessionId, useRAG]);

  // Toggle voice input
  const toggleVoiceInput = () => {
    if (!recognitionRef.current) {
      setVoiceStatus("Voice input not supported in this browser");
      setTimeout(() => setVoiceStatus(""), 3000);
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      setVoiceStatus("");
    } else {
      recognitionRef.current.start();
    }
  };
  // Create new session
  const createNewSession = () => {
    const newSessionId = `session_${Date.now()}_${Math.random()
      .toString(36)
      .substr(2, 9)}`;
    setSessionId(newSessionId);
    setVoiceStatus("New session started!");
    setTimeout(() => setVoiceStatus(""), 2000);
  };

  // Clear current session memory
  const clearCurrentSession = async () => {
    try {
      const response = await fetch(host + "/clear_session", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (response.ok) {
        setVoiceStatus("Session memory cleared!");
      } else {
        setVoiceStatus("Error clearing session");
      }
      setTimeout(() => setVoiceStatus(""), 2000);
    } catch (error) {
      console.error("Error clearing session:", error);
      setVoiceStatus("Error clearing session");
      setTimeout(() => setVoiceStatus(""), 2000);
    }
  };

  // Toggle mode
  const toggleMode = () => {
    setUseRAG(!useRAG);
    setVoiceStatus(
      useRAG ? "Switched to LLM-only mode" : "Switched to RAG mode"
    );
    setTimeout(() => setVoiceStatus(""), 2000);
  };

  // End of play
  function playerEnded(e) {
    setAudioSource(null);
    setSpeak(false);
    setPlaying(false);
    setIsThinking(false);
    setBlendData(null);
    setVoiceStatus("");
  }

  // Player is ready
  function playerReady(e) {
    audioPlayer.current.audioEl.current.play();
    setPlaying(true);
    setIsThinking(false);
    setVoiceStatus("");
  }

  return (
    <div className="full">
      <div style={STYLES.area}>
        <textarea
          rows={4}
          type="text"
          style={STYLES.text}
          value={text}
          onChange={(e) => setText(e.target.value.substring(0, 200))}
          placeholder="Type or speak your message..."
        />
        <button
          onClick={() => {
            setSpeak(true);
            setIsThinking(true);
          }}
          style={STYLES.speak}
          disabled={speak}
        >
          {speak ? "Running..." : "Ask"}
        </button>
        <button
          onClick={toggleVoiceInput}
          style={{
            ...STYLES.voiceButton,
            background: isListening ? "#ff4444" : "#444444",
          }}
          disabled={speak}
        >
          {isListening ? "🎤 Stop Listening" : "🎤 Voice Input"}
        </button>
        {voiceStatus && <div style={STYLES.status}>{voiceStatus}</div>}
      </div>

      {/* Control Panel - Top Right */}
      <div style={STYLES.area2}>
        <div style={STYLES.label}>Mode:</div>
        <button
          onClick={toggleMode}
          style={{
            ...STYLES.modeButton,
            background: useRAG ? "#2563eb" : "#dc2626",
          }}
        >
          {useRAG ? "🧠 RAG Mode" : "💭 LLM-only Mode"}
        </button>

        <div style={{ ...STYLES.label, marginTop: "15px" }}>Session:</div>
        <button onClick={createNewSession} style={STYLES.sessionButton}>
          🔄 New Session
        </button>
        <button onClick={clearCurrentSession} style={STYLES.sessionButton}>
          🗑️ Clear Memory
        </button>

        <div style={{ ...STYLES.label, marginTop: "10px", fontSize: "0.7em" }}>
          ID: {sessionId.substring(0, 15)}...
        </div>
      </div>

      <ReactAudioPlayer
        src={audioSource}
        ref={audioPlayer}
        onEnded={playerEnded}
        onCanPlayThrough={playerReady}
      />

      <Canvas
        dpr={2}
        onCreated={(ctx) => {
          ctx.gl.physicallyCorrectLights = true;
        }}
      >
        <OrthographicCamera makeDefault zoom={2000} position={[0, 1.65, 1]} />

        <Suspense fallback={null}>
          <Environment
            background={false}
            files="/images/photo_studio_loft_hall_1k.hdr"
          />
        </Suspense>

        <Suspense fallback={null}>
          <Bg />
        </Suspense>

        <Suspense fallback={null}>
          <Avatar
            avatar_url="/model.glb"
            speak={speak}
            setSpeak={setSpeak}
            text={text}
            setAudioSource={setAudioSource}
            playing={playing}
            isThinking={isThinking}
            blendData={blendData}
          />
        </Suspense>
      </Canvas>
      <Loader dataInterpolation={(p) => `Loading... please wait`} />
    </div>
  );
}

function Bg() {
  const texture = useTexture("/images/bg.webp");

  return (
    <mesh position={[0, 1.5, -2]} scale={[0.8, 0.8, 0.8]}>
      <planeGeometry />
      <meshBasicMaterial map={texture} />
    </mesh>
  );
}

export default App;
