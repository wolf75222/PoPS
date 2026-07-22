"""Vue live ParaView Catalyst 2 du tutoriel d'advection scalaire.

Le nom ``mesh`` correspond au canal du backend PoPS. ParaView charge ce fichier pendant
``catalyst.initialize``; la premiere frame acceptee fournit ensuite le champ ``U`` et son composant
``u`` declares par l'utilisateur.
"""

# script-version: 2.0

from paraview import catalyst
from paraview.simple import (
    ColorBy,
    CreateView,
    GetColorTransferFunction,
    ResetCamera,
    Show,
    TrivialProducer,
)


producer = TrivialProducer(registrationName="mesh")
view = CreateView("RenderView")
view.ViewSize = [1280, 720]
display = Show(producer, view, "UnstructuredGridRepresentation")

options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.CatalystLiveTrigger = "TimeStep"
options.EnableCatalystLive = 1

_configured = False


def catalyst_execute(info):
    """Apply presentation after the first PoPS frame has populated the producer."""
    del info
    global _configured
    producer.UpdatePipeline()
    if not _configured:
        display.SetRepresentationType("Surface With Edges")
        ColorBy(display, ("CELLS", "U"))
        color_map = GetColorTransferFunction("U")
        if color_map.ApplyPreset("Viridis", True) is False:
            raise RuntimeError("ParaView does not provide the Viridis color preset")
        display.RescaleTransferFunctionToDataRange(True)
        display.SetScalarBarVisibility(view, True)
        ResetCamera(view)
        _configured = True
