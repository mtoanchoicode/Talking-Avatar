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

import * as THREE from "three";
import axios from "axios";
import { SRGBColorSpace, LinearSRGBColorSpace } from "three";

const _ = require("lodash");

const host = "http://127.0.0.1:5000";

// Thinking animation data - subtle head nod and thinking expression
const thinkingData = [
  {
    time: 4.099999999999991,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0, // Start at 0
      mouthFrownRight: 0, // Start at 0
    },
  },
  {
    time: 4.116666666666657,
    blendshapes: {
      eyeBlinkLeft: 0.119,
      eyeBlinkRight: 0.119,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.05, // Slightly increase
      mouthFrownRight: 0.05, // Slightly increase
    },
  },
  {
    time: 4.133333333333324,
    blendshapes: {
      eyeBlinkLeft: 0.283,
      eyeBlinkRight: 0.283,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.1, // Increase
      mouthFrownRight: 0.1, // Increase
    },
  },
  {
    time: 4.149999999999991,
    blendshapes: {
      eyeBlinkLeft: 0.423,
      eyeBlinkRight: 0.423,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.15, // Increase
      mouthFrownRight: 0.15, // Increase
    },
  },
  {
    time: 4.166666666666657,
    blendshapes: {
      eyeBlinkLeft: 0.502,
      eyeBlinkRight: 0.502,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.2, // Max value
      mouthFrownRight: 0.2, // Max value
    },
  },
  {
    time: 4.183333333333324,
    blendshapes: {
      eyeBlinkLeft: 0.464,
      eyeBlinkRight: 0.464,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.18, // Decrease
      mouthFrownRight: 0.18, // Decrease
    },
  },
  {
    time: 4.19999999999999,
    blendshapes: {
      eyeBlinkLeft: 0.35,
      eyeBlinkRight: 0.35,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.14, // Decrease
      mouthFrownRight: 0.14, // Decrease
    },
  },
  {
    time: 4.216666666666657,
    blendshapes: {
      eyeBlinkLeft: 0.229,
      eyeBlinkRight: 0.229,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.1, // Decrease
      mouthFrownRight: 0.1, // Decrease
    },
  },
  {
    time: 4.233333333333324,
    blendshapes: {
      eyeBlinkLeft: 0.142,
      eyeBlinkRight: 0.143,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.06, // Decrease
      mouthFrownRight: 0.06, // Decrease
    },
  },
  {
    time: 4.24999999999999,
    blendshapes: {
      eyeBlinkLeft: 0.103,
      eyeBlinkRight: 0.103,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.04, // Decrease
      mouthFrownRight: 0.04, // Decrease
    },
  },
  {
    time: 4.266666666666657,
    blendshapes: {
      eyeBlinkLeft: 0.072,
      eyeBlinkRight: 0.072,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.02, // Decrease
      mouthFrownRight: 0.02, // Decrease
    },
  },
  {
    time: 4.2833333333333234,
    blendshapes: {
      eyeBlinkLeft: 0.04,
      eyeBlinkRight: 0.04,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.01, // Decrease
      mouthFrownRight: 0.01, // Decrease
    },
  },
  {
    time: 4.29999999999999,
    blendshapes: {
      eyeBlinkLeft: 0.022,
      eyeBlinkRight: 0.022,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.005, // Decrease
      mouthFrownRight: 0.005, // Decrease
    },
  },
  {
    time: 4.316666666666657,
    blendshapes: {
      eyeBlinkLeft: 0.012,
      eyeBlinkRight: 0.012,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0.002, // Decrease
      mouthFrownRight: 0.002, // Decrease
    },
  },
  {
    time: 4.333333333333323,
    blendshapes: {
      eyeBlinkLeft: 0.007,
      eyeBlinkRight: 0.007,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0, // Back to 0
      mouthFrownRight: 0, // Back to 0
    },
  },
  {
    time: 4.34999999999999,
    blendshapes: {
      eyeBlinkLeft: 0.005,
      eyeBlinkRight: 0.005,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.3666666666666565,
    blendshapes: {
      eyeBlinkLeft: 0.003,
      eyeBlinkRight: 0.003,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.383333333333323,
    blendshapes: {
      eyeBlinkLeft: 0.002,
      eyeBlinkRight: 0.002,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.39999999999999,
    blendshapes: {
      eyeBlinkLeft: 0.002,
      eyeBlinkRight: 0.002,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.416666666666656,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.433333333333323,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.4499999999999895,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.466666666666656,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.483333333333323,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.499999999999989,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.516666666666656,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.533333333333323,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.549999999999989,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.566666666666656,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.583333333333322,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.599999999999989,
    blendshapes: {
      eyeBlinkLeft: 0.001,
      eyeBlinkRight: 0.001,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.616666666666656,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.633333333333322,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.649999999999989,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.666666666666655,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.683333333333322,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.699999999999989,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.716666666666655,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.733333333333322,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.7499999999999885,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.766666666666655,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.783333333333322,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.799999999999988,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.816666666666655,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.8333333333333215,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.849999999999988,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
  {
    time: 4.866666666666655,
    blendshapes: {
      eyeBlinkLeft: 0,
      eyeBlinkRight: 0,
      mouthSmileLeft: 0.3,
      mouthSmileRight: 0.3,
      mouthFrownLeft: 0,
      mouthFrownRight: 0,
    },
  },
];

function Avatar({
  avatar_url,
  speak,
  setSpeak,
  text,
  setAudioSource,
  playing,
  isThinking,
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

  useEffect(() => {
    if (speak === false) return;

    makeSpeech(text)
      .then((response) => {
        let { blendData, filename } = response.data;

        let newClips = [
          createAnimation(blendData, morphTargetDictionaryBody, "HG_Body"),
          createAnimation(
            blendData,
            morphTargetDictionaryLowerTeeth,
            "HG_TeethLower"
          ),
        ];

        filename = host + filename;

        setClips(newClips);
        setAudioSource(filename);
      })
      .catch((err) => {
        console.error(err);
        setSpeak(false);
      });
  }, [speak]);

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
  const [text, setText] = useState(
    "My name is Arwen. I'm a virtual human who can speak whatever you type here along with realistic facial movements."
  );
  const [audioSource, setAudioSource] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("");
  const [isThinking, setIsThinking] = useState(false);

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

  // End of play
  function playerEnded(e) {
    setAudioSource(null);
    setSpeak(false);
    setPlaying(false);
    setIsThinking(false);
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
          {speak ? "Running..." : "Speak"}
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
