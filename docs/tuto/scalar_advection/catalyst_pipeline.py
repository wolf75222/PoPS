"""Pipeline live ParaView Catalyst 2 du tutoriel d'advection scalaire.

Le nom ``mesh`` correspond au canal du backend PoPS. ParaView charge ce fichier pendant
``catalyst.initialize``; la premiere frame acceptee fournit ensuite le champ ``U`` et son composant
``u`` declares par l'utilisateur. La pipeline reste volontairement sans ``RenderView`` : elle est
executee par le worker asynchrone PoPS et une vue Cocoa ne peut pas etre creee hors du thread
principal sur macOS. La presentation preconfiguree est portee par la recette/PVSM de la sortie
ParaView; le client Catalyst Live peut visualiser la source ``mesh`` pendant le calcul.
"""

# script-version: 2.0

import os

from paraview import catalyst
from paraview.simple import TrivialProducer


producer = TrivialProducer(registrationName="mesh")

options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.CatalystLiveTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveURL = os.environ.get("POPS_CATALYST_LIVE_URL", "localhost:22222")

def catalyst_execute(info):
    """Require the user-named field after PoPS has populated the live producer."""
    del info
    producer.UpdatePipeline()
    field = producer.GetCellDataInformation()["U"]
    if field is None or int(field.GetNumberOfComponents()) != 1:
        raise RuntimeError("Catalyst Live did not receive scalar cell field U")
