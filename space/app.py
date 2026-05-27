import gradio as gr
import time
import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

species_images = {
    "Micromonospora maris": "https://masscience.com/wp-content/uploads/2015/12/image_0068.jpg",
    "Streptomyces": "https://actinobase.org/images/thumb/e/e4/Streptomyces.png/300px-Streptomyces.png",
    "Streptomyces bottropensis": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSPuTt-opTY297E33iSfti8tLoFJg4lbZ4puA&s",
    "Cylindrospermopsis raciborskii T3": "https://inaturalist-open-data.s3.amazonaws.com/photos/12099755/large.jpg",
    "Paracoccus haeundaensis": "https://www.microbiologyresearch.org/docserver/ahah/fulltext/ijsem/54/5/IJE45760-1_thmb.gif",
    "Streptomyces viridochromogenes": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQdrOBXTeISHa7UCpAlchN-8eb-gl1ZJSi1CA&s",
    "Unknown": "https://static.thenounproject.com/png/3674270-200.png"
}

model_repo = "lm-jim/bgc-infiller"

tokenizer = AutoTokenizer.from_pretrained(model_repo, subfolder="bgc-infiller-esm2-v2.0")

model_v1_1 = AutoModelForMaskedLM.from_pretrained(model_repo, subfolder="bgc-infiller-esm2-v1.1")
model_v1_5 = AutoModelForMaskedLM.from_pretrained(model_repo, subfolder="bgc-infiller-esm2-v1.5")
model_v2_0 = AutoModelForMaskedLM.from_pretrained(model_repo, subfolder="bgc-infiller-esm2-v2.0")

def update_species_image(especie):
    url = species_images.get(especie, species_images["Unknown"])
    return url

def randomize_masking(sequence):
    if len(sequence) == 0:
        return ""

    tokens = sequence.split()
    tokens_to_mask = random.sample(range(len(tokens)), 5)
    for i in tokens_to_mask:
        if len(tokens[i]) == 1:
            tokens[i] = '<mask>'

    return " ".join(tokens)

def infill_protein(sequence, model, creativity):
    if "<mask>" not in sequence:
        return "Please, mask the amino acid sequence with the <mask> token."
    
    time.sleep(0.5)

    model_mapping = {
        "bgc-infiller-8M-v1.1": model_v1_1,
        "bgc-infiller-8M-v1.5": model_v1_5,
        "bgc-infiller-35M-v2.0": model_v2_0
    }
    
    selected_model = model_mapping[model]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_model.to(device)
    
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    print(inputs['input_ids'])
    masked_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        outputs = selected_model(**inputs)
        logits = outputs.logits
    
    print(logits)
    predicted_ids = inputs["input_ids"][0]
    
    for i in masked_indices:

        token_probability = logits[0, i]
        
        if creativity > 0:
            normalized_probs = torch.softmax(token_probability / (creativity / 10), dim=-1)
            predicted_token = torch.multinomial(normalized_probs, num_samples=1).item()
        else:
            predicted_token = token_probability.argmax(dim=-1).item()

        predicted_ids[i] = predicted_token
        
    infilled_sequence = tokenizer.decode(predicted_ids, skip_special_tokens=True)
    
    return infilled_sequence


with gr.Blocks(theme=gr.themes.Soft(), title="BGC Infiller 🧬") as demo:
    gr.Markdown("# 🧬 Biosynthetic Genomic Cluster (BGC) Infiller 🧬")
    gr.Markdown("A fine-tuned ESM2 model for infilling BGCs (Biosynthetic Genomic Clusters)")

    with gr.Row():
        input_species = gr.Textbox(
                label="Organism Species",
                value="Unknown", 
                interactive=False
            )
        input_bgc_class = gr.Textbox(
            label="BGC Class",
            value="Unknown", 
            interactive=False
        )
    with gr.Row():
        organism_img = gr.Image(
            label="Organism Photo",
            value=species_images["Unknown"],
            height=255,
            width=255,
            interactive=False,
        )

    with gr.Row():
        with gr.Column():
            input_seq = gr.Textbox(
                label="Masked BGC Sequence Input",
                placeholder="Enter your BGC sequence in the following format:\n\n[CLASS_TYPE] M K V L <mask> L A A I L\n\nAmino acids separated by one space. Maximum of 1000 amino acids or mask tokens.",
                lines=12
            )
            model_selector = gr.Radio(
                ["bgc-infiller-8M-v1.1", "bgc-infiller-8M-v1.5", "bgc-infiller-35M-v2.0"], 
                label="Model Selection", 
                value="bgc-infiller-35M-v2.0"
            )
            randomize_button = gr.Button("Randomize Masking 🎲", variant="primary")
            
        with gr.Column():
            output_display = gr.Textbox(label="Infilling Results", lines=12, interactive=False)
            creativity_selector = gr.Slider( 
                label="Model Creativity",
                minimum=0, 
                maximum=10,
                value=0
            )
            generate_button = gr.Button("Generate Infilled Protein Sequence 🦠", variant="primary")
            
    randomize_button.click(fn=randomize_masking, inputs=[input_seq], outputs=input_seq)
    generate_button.click(fn=infill_protein, inputs=[input_seq, model_selector, creativity_selector], outputs=output_display)

    gr.Examples(
        label="Select an existing organism BGC sequence",
        inputs=[input_species, input_bgc_class, input_seq],
        examples=[
            ["Micromonospora maris", "PKS", "[PKS] M S A P D E P V A V V G L A C R L P G A A D P E A F W A L L R D G R E A I T D P P A S R R D P D G R A R R G G F L D A V D L F D A E F F G V P P R E A A A M D P Q Q R L V L E L S W E A L E D A R I R P D A L A G S R T G V F V G A I S D D Y A T L L R R R G P D A I G P H S L T G T N R G I I A N R V S Y H L G L H G P S I T V D S A Q S S A L V A V H V A A E S L R R G E S E L A L A G G V N L N L A P E S T L G A E R F G A L S P D G R C H T F D A R A N G Y V R G E G G G L V V L K P L D R A L A D G D R V H A V L L G S A V N N D G V T D G L T V P G S D G Q R E V I R L A H E R A G T T P G E V D Y V E L H G T G T V V G D P V E A A A L G A E L G Q L R D T P L L V G S A K T N V G H L E G A A G I V G F V K A V L C V R H R T L P P S L N F A T P N P R I P L N E L N L R V V T E S H T V P R P L V V G V S S F G M G G T N A H A V L T E A P L R T A R K P A P A A R P L T W V V S G H T P Q A L R A Q A G Q L T S L A A D P A D V A F S L A T T R A T L P Y R A A V V G E T A A D L R A G M A A V A T G T P H P G T V T G S P A G T L A F V F T G Q G S Q R A G M G R E L A A R F P V Y A Q A F E V A A A L D P H L G R P L D E V L D D A D A L D R T E F A Q P A L F A V E V A L F R L L T H W G L R P D A V A G H S V G E I A A A H V A G V L D L P D A A R L V A A R G R L M Q A L P T G G A M V A L S A G E E E V R P L L R P G A D L A A V N A A E S V V V A G D D D A V S A I E E T V R G W G R R T S R L R V S H A F H S A R M D P M R V S F A Q A L A D I E F A Q P T I P V V S A L T D P D V T D A E H W V R H V R D T V R F A D A V R D L R D R G V R T V L E V G P D A V L T A L A H D V A E L A A V A V L R R D R P E P D T A V T A L A T A F T R G A A V D W T A L L G A R Q A V D L P R Y A F Q R S R H W L D Q D N P A I A A P E V T R A P D G T P R R S D E E L L D L V R T A V A V A H G R V G P A A I D P D T T F R D L G L D S V T S V E F R D R L A A A T G V P L S P G L V Y D H P T P R A V V A H L R T L T G G G P A D P E Q E S G Y R D E P V A V I G M A C R Y P G G V G S P D D L W Q L V R D G R D A T G P F P T D R G W D L D A L Y D P D P G T P G R T Y V R R G G F L D G A A E F D A D F F G I S P R E A S A M D P Q Q R L L L H T A W E A L E H G R L N P E S L R G T R T G V F V G V V D N D Y G P R L H E P V E G T E G Y L L T G T T A S V A S G R V A Y A L G L T G P A V T V D T A C S S S L V A L H L A A Q A L R Q G E C T L A L A G G A T V L A T P G M F L E F S R Q R G L A P D G R C K A F A A T A D G T A W A E G A G L V V L E R L S D A R R N G H P V L A V L R G S A I N Q D G A S N G L T A P S G P S Q E R V I R R A L A V A G L A P S D V D L M E A H G T G T A L G D P I E A R A I L A T Y G Q R R D T P L H L G S L K S N I G H T Q A A A G I A G V I K V V Q A M Q H G T L P A T L H V D E P T P H V D W A E G Q V S L L T E A T P W P D T G R P R R A A V S S F G I S G T N A H V I L E H G D P Q P A P P R R T S T G H V A W L V S A R E P E L V A E Q A G R L H R F V R D N P E L D P A D V A L S L A T T R P L L E H R A A V V G A D R D E L L A G L A E L E S G R R R A E A I R P G K V A F L F A Q G T Q R L N M G R Q L Y D T N P T F A H A L D T V T N A L N P H L N Q P L L D I I F G T D P H L L N R T E N A Q P A L F A I E T A L Y H L L T H H G I H P D Y L L G H S L G E I T A A H A A G I L T L T D A A T L V T T R A K L M Q T A T P G G A M I A I E A T E T E I Q P T L H P T V T I A A I N T P T T T V I S G D H H H T H A I A H H W R Q Q G R R T T T L T V S H A F H S P H M D P I L D T F H T T T Q T L T H H P P H T P L I T N L T G Q P L T N P T P E H W T H H L R Q P V R Y H D A T T T L T H H G V T H T I E I G P D T T L T T L T K T N H P T L T T T P T L R P H H N E N H T L T H T L A T T P T T N W A S L Y P H A R P V R L P T T A F R R D R Y W L T G G R A T P G A D S G V R E V D H P L L G A A V T L A D D S T V Y T G R L S R R T A P W L A D H V V L G R A L L P G T A L L E Y A L W A G R D V G L P R V A E L T L E A P L V L P D E G V T Q V R V T V G P P G E Q R T V A V H A R A D D A E Q W T R H A S G M L T A A P P A S P V P P R V D G P P V D V D D L Y E R L A G K G Y E Y G P A F R L A T A A R H G Q H V V A Q L A A P A G P D G F V L H P A Q V D A A L H P I V L D G D E T L L P F S W S G V S V F R R P S G A L H A Y W T P E R A L V L T D A D G V V A T A D S L H L R P A R M P A P T D L H R I R W V P A E D A R R Q I R V E P V A D A T A A L A M L H E R L D A T E P T A L V V P H L D R T G A A G L V R S A Q A E H P G R F V L I H A D D P V R T V P D G E P E A A W R D G S W W V P R L A R V A P V D P G L P L S G T V L V T G G T G A L G A L V A R H L V R A H R V R D L V L V S R R G A D A P G A A A L A D E L A G H G A R V D L R A C D V A D R E A L A C L L A D L P T L D A V V H A A G V V R D A T V S A L T V E Q V R A A A T K A E S A W H L H E L T R D R P L R A F V L F S S I S G L L G T A G Q G A Y A A A N A A L D A L A A H R H A L G L P A L S L A W G L W E D T G M G A G L S A A D V A R W R R D G L P P L T V E Q G V A L F D A A L S H E G P V L A P V R L D L A A L R G R D V L P A A L R G L V T R R A V P P A G S R P R D E A E L R E V V R S V V A E V L G Y P S A A G V D S A R P F R D L G L D S L G G V E L R N R L A A A T G L P V P A T L V F D H P T P D A V V A H L L G A T T S A Q P A P T P T V A T R T D E P I A I V G M A C R Y P G G V S S P E D L W R L V A D G V D A I G E F P T D R G W D L G R L Y D P D P E H A G T S Y T R H G G F L Y D A A D F D A G F F A L S P R E A T A T D P Q Q R L L L E V A W E A F E R A G I D P T A V R G S R T G V F A G V M Y G D Y G T R W R T A P E G F E G H L L T G N T S S V V S G R V A Y S F G L E G P A V T V D T A C S S S L V A L H L A A Q S L R S G E C D L A L A G G V T V M A T P H T F V E F S R Q R G L A P D G R C R S F S A A A N G T G W S E G A G L L L V E R L S D A R A N G H H V L A I L R G S A V N Q D G A S N G L T A P N G P A Q Q R V I R T A L T N A H L Q P T D I D L V E A H G T G T R L G D P I E A Q A L I A T Y G H H R N T P L H L G S L K S N I G H T Q A A A G V A G V I K V I Q A M Q H G T L P A T L H V N E P T P H V N W A D S Q V T L L T E A T P W P D T G R P R R A A V S S F G I S G T N A H V I L E H G D P R P V P P E E T D P P A P V P L V I S A R S A G A L R D Q A A R V R T A L G S G L P V R D V A Y T L G A A R A R H P H Q A V V V G E G R A E L L A G L D A V A D G T V P G A V A T P G K V A F L F A Q G T Q R L N M G R Q L Y D T N P T F A H A L D T V T N A L N P H L N Q P L L D I I F G T D P H L L N R T E N A Q P A L F A I E T A L Y H L L T H H G I H P D Y L L G H S L G E I T A A H A A G I L T L T D A A T L V T T R A K L M Q T A T P G G A M I A I E A T E T E I Q P T L H P T V T I A A I N T P T T T V I S G D H H H T H A I A H H W R Q Q G R R T T T L T V S H A F H S P H M D P I L D T F H T T T Q T L T H H P P H T P L I T N L T G Q P L T N P T P E H W T H H L R Q P V R Y H D A T T T L T H H G V T H T I E I G P D T T L T T L T K T N H P T L T T T P T L R P H H N E N H T L T H T L A T T P T T N W K T L L P H A T V I D L P T Y P F Q R Q R Y W L D G P A A D T G L D G S G H P L L P G V V D L A D G G L V L T G T V S A D S H P W L A G H R I G G A T L L P A T A V V E A V A H A A S R V G L D V D E L V L T A A V P V D A P V R L R L T V G P A S D D D S R A V H L H G N T D D G E W L P Y A T G R L A V V T Q S P A A D L A T W P P T D A E P V D V V D L Y D R L A E G G Y G Y H G L F Q G L R A L W R R G D E T F A E V R P D E S P T G G F A P H P A L W D A A L H P L A W D A A E R G Q V E I P F E W R S V R R H G P G A P A L R V R L A R R D D A V S V D V A D D A G R P I A S A G A L R L R S T G T A P T T V L E P D W E P V P T D G E W T G R Y A T V V A P R T G A D A S A A Y A A V T W A L D A L R Q H E G D E P L V V R T V D D P A G A A V R G L V R T A Q T E Q P G R F V L F T G S G D P E P A L V R A A L A S G E P E V A L R D G T L M A P R L S R I P V A P G P L P F A S G S T V L V T G G T G A L G A L V A R H L V V R H G V R R L L L T S R R G P A A D G A A E L V D E L T A A G A E V E V V A C D V A D R P A V A A L L A S I P E E H P L T A V I H T A G V L D D G A L T S L T E E R L A R V L R P K A E A A W H L H E F T R D R P L T A F V L F S S I T G I T G T A G Q A N Y A A A N A Y L D A L A R H R R N L G L P G V S L A W G L W G A T G M A S G L G A A D L D R L A R S G I T P L S P Q E G L D L F D A C L V A D R P V L A P A R V D L S T T R Q R R R A A S A A A T V T S R E G L R E L V R A Q V A A V L G H T D A T E V S T D V A F T G L G L D S L T A V E L R N R I A E R T G L R L S S T V V F D H P S V D A L T D H L V A E L A G A R P V E T P Q P V T Q P A D E P I A I V G M A C R Y P G G V S S P E D L W R L V A D G V D A I G E F P T D R G W G E I H D P D P D R P G H S Y T R H G G F L Y A A G D F D A E L F G M S P R E A L T T D P Q Q R L L L E V A W E A F E R A G L P P G S L R G S R T G V F T G V M Y N D Y G A R L H Q A G T P A P G Y E G Y L V S G S A G S V A S G R V A Y S F G L E G P A V T V D T A C S S S L V A L H L A A Q S L R S G E C D L A L A G G V T V M A S P A T F V E F S R Q R G L A P D G R C K P F A A A A D G T G W S E G A G L L L V E R L S D A R A N G H H V L A I L R G S A V N Q D G A S N G L T A P N G P A Q Q R V I R T A L T N A H L Q P T D I D L V E A H G T G T R L G D P I E A Q A L I A T Y G H H R N T P L H L G S L K S N I G H T Q A A A G V A G V I K V I Q A M Q H G T L P A T L H V N E P T P H V N W A D S Q V T L L T E A T P W P D T G R P R R A A V S S F G I S G T N A H V I L E H G D A V D T G T G T G T V A P G T A V V P W L L S G T S R Q A L T A Y A R L L G E V D A A P V D I A A T L A L G R S P L A L R A S V V G R D R P E F P T A R L Q P V Q P V D G P T A F A F T G Q G S Q R A G M G L G L A A R F P Q F A D A L A S V A E A L D P H L P S P L L D V L A D G D L L E R T E Y A Q P A I F A V E V A L F R L L A H Y G V T P H V L L G H S V G E L A A A H V A G V L D L P D A A T L V A A R G R L M G G L V P G G A M A A V R A G E D E V L A L L V P G A E I A A V N A D D A V V V S G D A E A V A A V T Q A L R D A G R R V T P L R V S H A F H S A R M D P V L E E F R A V A A T L R F S E P T I P L I S L L P G S P T D P G Y W V R H L R E A V R F G D G V R S L A E W G V R R V L E V G P D A A L T P V T G P T G I A T L R R D H D E E S A F V T A L A A L H D T G A T V D W A T F F G E L G A R R V P L P T Y P F Q R R R Y W L T P T A P R T T G G S G H P L L D A A V E L P E G A V L F T G R V A A E D A D W L A D H V V L G Q T V V S G A T L L S L V L H A A A A A G R P T V R R L T L H A P L V L P D D G G A A D L R V G V D E Q G Q V T V Y A R P A G G G W T R H A S G T L D T V E Q P A E A L G S W P P A G A E P L D V D Y T R L A D A G Y A Y G P G C R R V R A A W R L G D D L Y A E V G P V D A D G H A A P H P A L L D A A L H P L A L D L L D D E Q T R V P H V W S S V T V H A T G A T T L R A R I R R T G T D R V A L T L T D T D D R P V A T A D L T V R A V A R G L P D L Y A V R L T P V R P A T G G T V W P S V G R D V G L P R Y A E L S T S T D D I V E R A H D R V T E V A E L L R R W L A Q G P P E A R L V V A T D Q V T D P A D G V L W G L V R A A Q T E H P D R F V L L D S D G D P R S R T L V P G A L A T G E P Q L V V R D G R I T V P R L A R T A A A P Q P P R L D P D G T V L V T G A G G A L G S L T A R R L V T H H G V R R L L L L G R R G G M Q P L A A E L T A L G A T V R V A A C D A A N R A A L A R V L D T V P A A H P L T A V V H A A G V V S D G P L A T L T P Q R F A E V L R P K V D A A W H L H E L T C E Q D L A A F V L F S S L A G L V G N A G Q A N Y A A A N T G L D A L A A Y R R A A G L P A V S L A W G L W D A P G M G A A L D E T Q R A R I A R T G V A P L P V E R G L A L F D A C L G A R E A L L V P A A L Q P E R A T R V A P V L A G L A P A T T A T T P Q Q D W P R R L A G R G A A E Q H R L L L E L V R S T I V E V L G H S S V A A V A P D R G L M D L G F D S L T A V E L A G R L G A D T G V R T P S T V V F D H P T P T A L A H Y L R H E L V G E E A A D D E K P H E L D E V S D E D L F A L I D T E L G E R"],
            ["Streptomyces", "NRPS", "[NRPS] M V P V H A H D Y V T D P P S T T G R T L D G L T L P R V F A D A V H R G G D A V A L V D G E Y A L T W S A W R T A V D A L A R G L Q E S G V V S G D V V A L H L P N S W E Y L T L H L A A A S V G A V T M P V H Q G N A P S D V R A L L E R V R P A A V V L P A R T Q E G G G P L T G T A L R E V L P E L R A V L V T G D A A G E G T E T V T E M L E R W S G E D P L P V E V R P D S P F L L L P S S G T T S A R P K I C L H S H E G L L T N S R A A T E D T A D A Y A G T L I T A C P L T H C F G L Q S A Y S A L F R A G R Q V L L S G W D V G R F L E L A R R E R P S V V V A V P A Q L H D L V T R V R E D A D G P G F R P G R I L T A G A A L P P A L V R D V R E A L D T T L V V V W G M S E A G N G T S S L S A D A P E V V S R S V G R P T R D A E M R V V D E D G A P C P P G Q P G E L Y Y R S P S M F R G Y F G E P E L T R S V V S E D G W L R T G D L A S I G E D G L V T F H G R S A E L I N V G G R K F N A V E I Q A L L A D L P D I G P L A V V A A P D P R L G E Y P V L V V T E R P A A A P A D G T A P R P R G T V G L D E V T A H L R G L G T A E Y K I P L E L V A L P E L P R T P A G K I N R R A L E Q Y L A D A A E R T A V T P A E A P R P G L R T A L E L V V T A V A E V L A A V P G E D G A R P A A A G P I G P D T T F R A H G L D S V A S V R L R N A L A E A T G L T L P A G L A F D F P T P A A L A R E L A G L S S P A A E E S P G A S A H E D E P V A I V S M A C R L P G G A T S P E A L W E L L R D G V D A V S G F P E D R G W D L D A L F G D D P D A P G T S V A R E G G F L R D A A H F D A G F F G M S A R E A L A T D P Q Q R L L L E T A W E A V E R A G I A P R T L R G S R T G V F T G A M Y H D Y A A G A S D P A G E L E S L L P V G T A G G A L S G R I A Y T L G L S G P A L T V D T A C S S S L V A L H L A C R S L R S G E S D L A L A G G V A V M A T P A A F V G F S R L R G L S P D G R C K S F G E G A D G A A W S E G A G L L M L E R L S D A R R N G H P V L A V I R G S A V N Q D G A S N G L T A P H G P A Q R R V V R Q A L A D A G V R A A E V D V V E A H G T G T A L G D P I E A E A L L D T Y G R D R P E G R P L W L G S V K S N L G H T Q A A A G A A A V I K M V L A L R H D L L P A T L H A D T P T S R V D W S P G T V Q L L T R A R D W P R E E G R P R R A G V S S F G I S G T N A H L V L E E A P V P A A G T E R S A D A G A A G L R A A V P W L V S A K D A D A L R G Q A R R L A A H A A A H P E V S A R D L A Y S L L T T R A L H P R T A L L T G G D R D A L V A S A D A F A R G E A P G S I V R G P L G P A P G T A F V L T G Q G S Q R L G M G R G L A A A F P V F D D A L R E V C A L L D P L L E R P L T E V M W A A P D S D E A G L L G G T G Y A Q P A L F A F E V A L Y R L L E S W G I V P D R L V G H S V G E I A A A H V A G V L S L P D A C A L V A A R G R L M Q A L P P G G A M A A V R C S E A E I L P L L A G R T A G A T V A A V N G P R S V V L S G T E E A V A E V V T E V S A A G H K T R R L M V S H A F H S P L M E P M L A E F R A T V A G L S F A A P Q V P L V S G V T G R P L T A E E A R D P D H W V R H A R D T V R F A D A I S H L A G E H T E I Y V E L G P E A A L T P M V E E C L G E P E S G D G P A V E P V V R G D V D E E R A A L A A A V R L H A L G L D V Q W R A V L P E A R A V P L P T Y A F Q H E A Y W L A T S G S V V A G L S L P G G R A A D T V P D L A G R L A G L S G G E A E A L V T E L V R T E L A A V T G G E I S A A G A G T A F T E L G V T S V T A V E L R N R L T A V T G V R L P P T L I F D H P T P T A V A R L I G E T V R G S S V P G R R D A V S L V D E L E A L L V S G A E V D S D T A A R L R S L A G R W A P S A T G T A A D A N G P L D L D D A S D E E L F R L M D G G A P"],
            ["Streptomyces bottropensis", "Ribosomal", "[RIBOSOMAL] M V S R D G T P I R G F S R P G P G E T L V L V H G V A M D R R I W A E S G F L D A V P D A H V L A L D L R G R G E S G R V G T A R G H A L A R Y V E D V R A V L D G F G L A R Y S L F G T F F G G R I A L Q T A A V D P R V V R A F S F C A H A E Q V E I P E D A V E E E A V A V E G P G G H A Y L R D H F T G R G A P P W M V A A C A R V D P G E L G A A T R G L L H G S D R R T E R G H P D Q E L V L I T A D G D A D L A P F H A G E H R L G A R L W L V G A P T R I R A A G R L A E T G R R V A D V L A G T G R G D E D A G V E P G A G T A R A G R G G M T A T G T T A T E D T M W R R R R G L L P E G D P L V R A L E E P A G R G R G H R L R A V C L I D P A T G R P G P R T S"],
            ["Cylindrospermopsis raciborskii T3", "Other", "[OTHER] M Q I L G I S A Y Y H D S A A A M V I D G E I V A A A Q E E R F S R R K H D A G F P T G A I T Y C L K Q V G T K L Q Y I D Q I V F Y D K P L V K F E R L L E T Y L A Y A P K G F G S F I T A M P V W L K E K L Y L K T L L K K E L A L L G E C K A S Q L P P L L F T S H H Q A H A A A A F F P S P F Q R A A V L C L D G V G E W A T T S V W L G E G N K L T P Q W E I D F P H S L G L L Y S A F T Y Y T G F K V N S G E Y K L M G L A P Y G E P K Y V D Q I L K H L L D L K E D G T F R L N M D Y F N Y T V G L T M T N H K F H S M F G G P P R Q A E G K I S Q R D M D L A S S I Q K V T E E V I L R L A R T I K K E L G V E Y L C L A G G V G L N C V A N G R I L R E S D F K D I W I Q P A A G D A G S A V G A A L A I W H E Y H K K P R T S T A G D R M K G S Y L G P S F S E A E I L Q F L N S V N I P Y H R C V D N E L M A R L A E I L D Q G N V V G W F S G R M E F G P R A L G G R S I I G D S R S P K M Q S V M N L K I K Y R E S F R P F A P S V L A E R V S D Y F D L D R P S P Y M L L V A Q V K E N L H I P M T Q E Q H E L F G I E K L N V P R S Q I P A V T H V D Y S A R I Q T V H K E T N P R Y Y E L I R H F E A R T G C A V L V N T S F N V R G E P I V C T P E D A Y R C F M R T E M D Y L V M E N F L L V K S E Q P R G N S D E S W Q K E F E L D"],
            ["Paracoccus haeundaensis", "Terpene", "[TERPENE] M T H D V L L A G A G L A N G L I A L A L R A A R P D L R V L L L D H A A G P S D G H T W S C H D P D L S P H W L A R L K P L R R A N W P D Q E V R F P R H A R R L A T G Y G S L D G A A L A D A V A R S G A E I R W N S D I A L L D E Q G A T L S C G T R I E A G A V L D G R G A Q P S R H L T V G F Q K F V G V E I E T D C P H G V P R P M I M D A T V T Q Q D G Y R F I Y L L P F S P T R I L I E D T R Y S D G G N L D D D A L A A A S H D Y A R Q Q G W T G A E V R R E R G I L P I A L A H D A A G F W A D H A E G P V P V G L R A G F F H P V T G Y S L P Y A A Q V A D V V A G L S G P P G T D A L R G A I R D Y A I D R A R R D R F L R L L N R M L F R G C A P D R R Y T L L Q R F Y R M P H G L I E R F Y A G R L S V A D Q L R I V T G K P P I P L G T A I R C L P E R P L L K E N A"],
            ["Streptomyces viridochromogenes", "Saccharide", "[SACCHARIDE] M V V A V C A F R L E N V R R H L R H N L D Q L N G D E Y V V L L D R P V T P E A E K V A T Q V N E A G G T M R I L G A T R G L S A S R N T V L R E W A D R H V L F V D D D V R L E A S A V D A V R A A F R A G A H V V G A R L R P P R E L R R L P W F L S S G Q F H L V G W H R D R G D I K I W G A C M G V D A D F A R R Q G L T F D L D L S R T G V N L Q S G E D T S F I A L M K E A G A R E L P A A R A R G R P R C R P R P A H P P L P P A P G L L A G C V R R R D G T S R R R D S A R S"]
        ]
    )

    input_species.change(
        fn=update_species_image,
        inputs=[input_species],
        outputs=[organism_img]
    )

if __name__ == "__main__":
    demo.launch()