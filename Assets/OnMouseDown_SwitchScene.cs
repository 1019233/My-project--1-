using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class OnMouseDown_SwitchScene : MonoBehaviour
{
    public string sceneName = "";

    public void MouseDown()
    {
        if (sceneName != "")
        {
            SceneManager.LoadScene(sceneName);
        }
        else
        {
            int nextIndex = SceneManager.GetActiveScene().buildIndex + 1;
            if (nextIndex < SceneManager.sceneCountInBuildSettings)
            {
                SceneManager.LoadScene(nextIndex);
            }
            else
            {
                SceneManager.LoadScene(0);
            }
        }
    }

}
